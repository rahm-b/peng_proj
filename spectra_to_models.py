# Module to train classifiers based on a dictionary of spectra

import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import pandas as pd

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


particle_types = ['Er0p5Tm02Yb97p5', 'Er01Tm02Yb20_NEW_PMAO', 'Er01Tm05Yb94_NEW_PMAO', 'Er02Tm01Yb97_NEW_PMAO',
                  'Er02Tm05Yb93', 'Er02Yb20', 'Er04Yb13', 'Tm01Yb99', 'Tm02Yb98', 'Tm10Yb90']


# Now, let's organize our data. Currently, we have pkl files which consist of dictionaries within a dictionary.
# Current: specific particle ID --> frame avg --> list of two arrays (wavelegth and avg counts) (both arrays have length 100)
# What we want new: one big dictionary that contains everything


# # First create a dictionary with everything we need
# full_dict = {}

# for particle_type in particle_types:
#     with open(f"database/{particle_type}_spectra_cleaned_background substracted_should be singles I hope_1.pkl", "rb") as f:   # "rb" = read binary
#         data_ptcl = pickle.load(f)
#         full_dict[particle_type] = data_ptcl

# get all the spectra for a certain particle type and certain frame average
def get_all_spectra_ptcl(particle_type, frame_avg, full_dict):
    list_of_spectra = []

    for this_id_dict in full_dict[particle_type].values():
        list_of_spectra.append(this_id_dict[frame_avg])
    
    return list_of_spectra

# returns a list of spectra. Each element is its own tuple list 
# consisting of two 100-length numpy arrays (for wavelengths and avg counts)



# get the spectrum for a certain particle id and frame avg
def get_spectrum_from_id(particle_type, particle_id, frame_avg, full_dict):
    particle_dict = full_dict[particle_type][particle_id]
    spectrum = particle_dict[frame_avg]
    return spectrum


# function to plot all the recorded spectra for a certain particle type (can either choose normalized or not normalized)

def graph_particle_type(particle_type, frame_avg, full_dict, normalized=False):
    isNormalizedString = "" if not normalized else " (normalized)"

    # get list of relevant spectra
    list_of_spectra = get_all_spectra_ptcl(particle_type, frame_avg, full_dict)
    for spectrum in list_of_spectra:
        wavelengths = spectrum[0]
        counts = spectrum[1]

        # if normalized, we want to divide counts by total area
        if normalized:
            # Compute integral
            area = np.trapezoid(counts, wavelengths)

            # Normalize counts
            counts = counts / area  

        plt.plot(wavelengths, counts)
    

    plt.title(f"{particle_type} with {frame_avg} frame average {isNormalizedString}")
    plt.show()


# Now, we want to write functions to create wavelength grids. 
# First, we need a function that gets the largest minimum and the smallest maximum 
# value among each frame average (so that we only use interpolation, not extrapolation 
# in our projections). Then, we can create grids.


def get_min_and_max(frame_avg, full_dict):
    minimums = []  # list of the minimum wavelength for each particle spectrum
    maximums = []  # list of the maximum wavelength for each particle spectrum

    for particle_type in particle_types:
        spectra = get_all_spectra_ptcl(particle_type, frame_avg, full_dict)
        for spectrum in spectra:
            wavelengths = spectrum[0]
            minimums.append(min(wavelengths))
            maximums.append(max(wavelengths))

    # we want the largest minimum (rounded to the smallest integer greater than it)
    # and smallest maximum (rounded to the largest integer smaller than it)
    min_wavelength = round(max(minimums) + 0.5)  # e.g. if max(minimums) = 440.1, we want min_wavelength to be 441
    max_wavelength = round(min(maximums) - 0.5)  # e.g. if min(maximums) = 891.9, we want max_wavelength to be 891

    return min_wavelength, max_wavelength

# IMPORTANT NOTE: THE MIN AND MAX ARE THE SAME FOR ALL FRAME AVERAGES!!! ALWAYS (443, 884)



# function to create the a grid. Input for grid_type can either be 'unif' or 'scaled'

def create_grid(frame_avg, grid_type, full_dict):

    min_wavelength, max_wavelength = get_min_and_max(frame_avg, full_dict)

    # if grid_type is 'unif'
    if grid_type == "unif":
        uniform_grid = np.linspace(min_wavelength, max_wavelength, max_wavelength - min_wavelength + 1)
        return uniform_grid, min_wavelength, max_wavelength
    
    # otherwise (scaled grid)
    # get min and max wavelength intervals
    min_interval_avg = np.mean([
        wavelengths[1] - wavelengths[0]  # the smallest interval is the first interval (difference between first and second wavelength datapoint)
        for particle_type in particle_types  # getting the average of all the minimum intervals for each particle_type
        for wavelengths in [spectrum[0] for spectrum in get_all_spectra_ptcl(particle_type, frame_avg, full_dict)]   # averaging over all particle_types
    ])
    max_interval_avg = np.mean([
        wavelengths[-1] - wavelengths[-2]  # the largest interval is the last interval
        for particle_type in particle_types  # getting the average of all the maximum intervals for each particle_type
        for wavelengths in [spectrum[0] for spectrum in get_all_spectra_ptcl(particle_type, frame_avg, full_dict)]   # averaging over all particle_types
    ])
    
    # now create the correct wavelength grid
    slope = (max_interval_avg - min_interval_avg) / (max_wavelength - min_wavelength)
    b = min_interval_avg - slope * min_wavelength

    current_wavelength = min_wavelength
    wavelengths = [min_wavelength]
    while True: # generate 100 wavelengths with the correct interval spacing
        low_interval = slope * current_wavelength + b
        new_wavelength = current_wavelength + low_interval
        high_interval = slope * new_wavelength + b
        avg_interval = (low_interval + high_interval) / 2
        current_wavelength += avg_interval
        if current_wavelength > max_wavelength:
            break
        wavelengths.append(current_wavelength)

    return np.array(wavelengths), min_wavelength, max_wavelength


"""Now, we want to create CSV files with our labeled data that we can use for our classifier. 
For each particle, its spectrum will be represented with a vector. Each element of the vector 
should be a normalized count value corresponding to a certain wavelength. The structure of the 
CSV is as follows (the header is shown):

`|   Particle id   |   Particle type   |   Spectrum vector (unif, 1 frame)   |   Spectrum vector (scaled, 1 frame)   |   Spectrum vector (unif, 2 frames)   |   Spectrum vector (scaled, 2 frames)   |   ...   |   Spectrum vector (scaled, 200 frames)   |`"""


def create_complete_csv_data(min_wavelength, max_wavelength, frame_avgs_list, full_dict, csv_name):

    # create the CSV and write the header
    csv_title = f"{csv_name}_{min_wavelength}to{max_wavelength}.csv"
    with open(csv_title, "w", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(["Particle id", "Particle type", 
        #                  "Spectrum vector (unif, 1 frames)", "Spectrum vector (scaled, 1 frames)",
        #                  "Spectrum vector (unif, 2 frames)", "Spectrum vector (scaled, 2 frames)",
        #                  "Spectrum vector (unif, 5 frames)", "Spectrum vector (scaled, 5 frames)",
        #                  "Spectrum vector (unif, 10 frames)", "Spectrum vector (scaled, 10 frames)",
        #                  "Spectrum vector (unif, 20 frames)", "Spectrum vector (scaled, 20 frames)",
        #                  "Spectrum vector (unif, 50 frames)", "Spectrum vector (scaled, 50 frames)",
        #                  "Spectrum vector (unif, 100 frames)", "Spectrum vector (scaled, 100 frames)",
        #                  "Spectrum vector (unif, 200 frames)", "Spectrum vector (scaled, 200 frames)"])  # header row
        header = ["Particle id", "Particle type"]

        for f in frame_avgs_list:
            header.append(f"Spectrum vector (unif, {f} frames)")
            header.append(f"Spectrum vector (scaled, {f} frames)")

        writer.writerow(header)
    

    # establish our two wavelength grids (unif and scaled)
    unif_grid = create_grid(1, 'unif', full_dict)[0]  # the frame_avg doesn't actually matter. Any will produce the same result.
    scaled_grid = create_grid(1, 'scaled', full_dict)[0]  # the frame_avg doesn't actually matter. Any will produce the same result.


    # loop through all spectra of every single particle (by looping through all particle types, particle ids, and frame avgs)
    for particle_type in particle_types:
        for particle_id in full_dict[particle_type].keys():
            this_ptcl_spectrum_vec_dict = {}  # create a dictionary for the particle's spectrum vector for different grid types and frame averages
            this_ptcl_spectrum_vec_dict['unif'] = {}
            this_ptcl_spectrum_vec_dict['scaled'] = {}

            for frame_avg in frame_avgs_list:
                this_ptcl_spectrum = get_spectrum_from_id(particle_type, particle_id, frame_avg, full_dict)
                wavelengths = this_ptcl_spectrum[0]
                counts = this_ptcl_spectrum[1]

                # project the counts onto the two wavelength grids: uniform grid and scaled grid
                counts_unif_grid = np.interp(unif_grid, wavelengths, counts)
                counts_scaled_grid = np.interp(scaled_grid, wavelengths, counts)

                # normalize these counts
                norm_counts_unif_grid = counts_unif_grid / np.trapezoid(counts_unif_grid, unif_grid)
                norm_counts_scaled_grid = counts_scaled_grid / np.trapezoid(counts_scaled_grid, scaled_grid)

                # add these normalized counts (vectors that represent the spectrum) to our spectrum_vec dictionary
                this_ptcl_spectrum_vec_dict['unif'][frame_avg] = norm_counts_unif_grid
                this_ptcl_spectrum_vec_dict['scaled'][frame_avg] = norm_counts_scaled_grid
            
            # now, add all the data from this specific particle (id) as a row in the CSV (we add rows to CSV one by one)
            with open(csv_title, "a", newline="") as f:
                writer = csv.writer(f)
                # writer.writerow([particle_id, particle_type,
                #                  this_ptcl_spectrum_vec_dict['unif'][1], this_ptcl_spectrum_vec_dict['scaled'][1],
                #                  this_ptcl_spectrum_vec_dict['unif'][2], this_ptcl_spectrum_vec_dict['scaled'][2],
                #                  this_ptcl_spectrum_vec_dict['unif'][5], this_ptcl_spectrum_vec_dict['scaled'][5],
                #                  this_ptcl_spectrum_vec_dict['unif'][10], this_ptcl_spectrum_vec_dict['scaled'][10],
                #                  this_ptcl_spectrum_vec_dict['unif'][20], this_ptcl_spectrum_vec_dict['scaled'][20],
                #                  this_ptcl_spectrum_vec_dict['unif'][50], this_ptcl_spectrum_vec_dict['scaled'][50],
                #                  this_ptcl_spectrum_vec_dict['unif'][100], this_ptcl_spectrum_vec_dict['scaled'][100],
                #                  this_ptcl_spectrum_vec_dict['unif'][200], this_ptcl_spectrum_vec_dict['scaled'][200]])
                row = [particle_id, particle_type]
                for f in frame_avgs_list:
                    row.append(this_ptcl_spectrum_vec_dict['unif'][f])
                    row.append(this_ptcl_spectrum_vec_dict['scaled'][f])
                writer.writerow(row)
                

"""Let's now create pipelines for different types of classifier models. 
The common types of classifiers we will use are:
- logistic regression
- SVM
- MLP
- Random Forest (RF)

Let's write functions for the pipelines for each type of classifier.

FIX THESE TO ALLOW FOR HYPERPARAMETRIZATION AND OPTIMIZATION!!! S:DLKJFS:DLKJDS:"""


# Map names → classifier classes
CLASSIFIERS = {
    "log_reg": LogisticRegression,
    "svm": SVC,
    "mlp": MLPClassifier,
    "rf": RandomForestClassifier
}


"""Now, let's write a function that allows us to train a model using our CSV and one of the above pipeline functions."""


# training for any type of model
def train_model_from_csv(frame_avg, grid_type, model_type, csv_path, **hyperparams):
    """
    model_type: str — one of ["log_reg", "svm", "rf"]
    hyperparams: passed directly to the model constructor
    """

    # CSV_PATH = f"all_data_labeled_443to884.csv"
    CSV_PATH = csv_path
    LABEL_COL = "Particle type"
    SPEC_COL  = f"Spectrum vector ({grid_type}, {frame_avg} frames)"

    df = pd.read_csv(CSV_PATH)

    # Parse spectra
    X_list = [np.fromstring(s.strip("[]"), sep=" ") for s in df[SPEC_COL].values]
    X = np.vstack(X_list)  # shape: (n_samples, n_features)

    # Encode labels
    le = LabelEncoder()

    y = le.fit_transform(df[LABEL_COL].values)

    # ---- Train / Test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type not in CLASSIFIERS:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # instantiate classifier with hyperparameters
    clf_type = CLASSIFIERS[model_type](**hyperparams)
    
    # Example pipeline (optional)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf_type)
    ])

    clf.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)

    # evaluate on the training data _______________
    y_train_pred = clf.predict(X_train)
    acc_train = accuracy_score(y_train, y_train_pred)

    classes = le.classes_

    joblib.dump(le, "label_encoder.pkl")

    return {"clf": clf,
            "frame_avg": frame_avg,
            "accuracy": acc, 
            "balanced_accuracy": bacc,
            "acc_train": acc_train,
            "classes": classes,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "classification_report": classification_report(y_test, y_pred, target_names=le.classes_),
            }


# Function to visualize and plot the results from the model trained in the function above
def plot_results(classes, y_test, y_pred, frames, acc, bacc, acc_train):
    print(f"Training accuracy: {acc_train:.3f}")
    # ___________________________________-

    print(f"Accuracy:          {acc:.3f}")
    print(f"Balanced accuracy: {bacc:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)  # room + smart layout

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=classes, ax=ax, colorbar=True, normalize="true"
    )
    int_classes = np.arange(len(classes))
    cm = confusion_matrix(y_test, y_pred, labels=int_classes)

    # Make sure ticks are at integer centers and labels align nicely
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_title(f"Number of frames: {frames}")

    # If labels still get clipped, add a little bottom margin:
    plt.subplots_adjust(bottom=0.22)
    plt.show()

    return fig, cm


# Whole pipeline in one function
def classifier(frame_avg, grid_type, model_type, csv_path, plot_the_results=False, show_incorrect=False, **hyperparams):  # grid_type is either "unif" or "scaled"
    trained_model = train_model_from_csv(frame_avg, grid_type, model_type, csv_path, **hyperparams)
    if plot_the_results:
        plot_results(trained_model['classes'], trained_model['y_test'], trained_model['y_pred'],
                     trained_model['frame_avg'], trained_model['accuracy'], trained_model['balanced_accuracy'], trained_model['acc_train'])
    return trained_model


