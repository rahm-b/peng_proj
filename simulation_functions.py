import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import poisson

particle_types = ['Er0p5Tm02Yb97p5', 'Er01Tm02Yb20_NEW_PMAO', 'Er01Tm05Yb94_NEW_PMAO', 'Er02Tm01Yb97_NEW_PMAO',
                  'Er02Tm05Yb93', 'Er02Yb20', 'Er04Yb13', 'Tm01Yb99', 'Tm02Yb98', 'Tm10Yb90']

def get_min_and_max(frames, particle_type):
    minis = []
    maxis = []
    with open(f"database/{particle_type}_spectra_cleaned_background substracted_should be singles I hope_1.pkl", "rb") as f:   # "rb" = read binary
        data = pickle.load(f)
    for particle in data.keys():
        wavelengths = data[particle][frames][0]
        minis.append(min(wavelengths))
        maxis.append(max(wavelengths))

    my_min_wavelength = round(max(minis) + 0.5)
    my_max_wavelength = round(min(maxis) - 0.5)

    return my_min_wavelength, my_max_wavelength

def create_unif_grid(frames, particle_type):
    min_wavelength, max_wavelength = get_min_and_max(frames, particle_type)
    uniform_grid = np.linspace(min_wavelength, max_wavelength, 10 * (max_wavelength - min_wavelength + 1))
    # return uniform_grid, min_wavelength, max_wavelength

    # also get scaled grid
    # get min and max intervals
    with open(f"database/{particle_type}_spectra_cleaned_background substracted_should be singles I hope_1.pkl", "rb") as f:   # "rb" = read binary
        data = pickle.load(f)
    example_particle = list(data.keys())[0]
    wavelengths = np.array(data[example_particle][frames][0])
    min_index = (np.abs(wavelengths - min_wavelength)).argmin()
    max_index = (np.abs(wavelengths - max_wavelength)).argmin()
    # get diff
    min_interval = wavelengths[min_index + 1] - wavelengths[min_index]
    max_interval = wavelengths[max_index] - wavelengths[max_index - 1]
    
    # now create the correct wavelength grid
    slope = (max_interval - min_interval) / (max_wavelength - min_wavelength)
    b = min_interval - slope * min_wavelength

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
    scaled_grid = np.array(wavelengths)

    return uniform_grid, scaled_grid, min_wavelength, max_wavelength


# first, we need to get the "ground truth"-ish spectra. We'll use the 200 frame averages.
# using several different "ground truths" for each particle (just using the 200-frame averages as the several possible ground truths)
def get_true_spectrum_new_varying_random(particle_type):
    list_of_counts_on_grid = []
    list_of_wavelengths = []
    list_of_counts = []
    norm_list_of_counts = []

    # first create uniform grid to project all our 200-frame average counts for each particle onto
    uniform_grid, scaled_grid, min_wavelength, max_wavelength = create_unif_grid(200, particle_type)
    with open(f"database/{particle_type}_spectra_cleaned_background substracted_should be singles I hope_1.pkl", "rb") as f:   # "rb" = read binary
            data = pickle.load(f)
    for particle in data.keys():
        wavelengths = data[particle][200][0]
        counts = data[particle][200][1]
        # project the counts onto the uniform grid
        counts_on_grid = np.interp(uniform_grid, wavelengths, counts)
        norm_counts = counts / np.trapezoid(counts, wavelengths)
        list_of_counts_on_grid.append(counts_on_grid)
        list_of_wavelengths.append(wavelengths[(wavelengths >= min_wavelength) & (wavelengths <= max_wavelength)])
        list_of_counts.append(counts[(wavelengths >= min_wavelength) & (wavelengths <= max_wavelength)])
        norm_list_of_counts.append(norm_counts[(wavelengths >= min_wavelength) & (wavelengths <= max_wavelength)])
    
    # Choose a random one
    random_chosen_counts = np.array(list_of_counts_on_grid[np.random.randint(len(list_of_counts))])

    # normalize
    norm_random_counts_unif_grid = random_chosen_counts / np.trapezoid(random_chosen_counts, uniform_grid)

    return uniform_grid, norm_random_counts_unif_grid, list_of_wavelengths, norm_list_of_counts



# plot a histogram of the number of total counts (intensities) of spectra
frame_count = 200
particle_counts_dict = {}

frame_counts = [1,2,5,10,20,50,100,200]

# we want to get the approx distribution of the brightnesses of each particle type so 
# that we can simulate particles with different brightnesses/counts. We will find the mean
# and standard deviation and approximate the distributions as Gaussian. I showed that a Gaussian
# is pretty close to accurate for most of these particles in a previous notebook.
for particle_type in particle_types:
    particle_counts_dict[particle_type] = {}
    particle_counts = []
    with open(f"database/{particle_type}_spectra_cleaned_background substracted_should be singles I hope_1.pkl", "rb") as f:   # "rb" = read binary
        data = pickle.load(f)
    for frame_num in frame_counts:
        for particle in data.keys():
            wavelengths = data[particle][frame_num][0]
            counts = data[particle][frame_num][1]
            total_counts = np.trapezoid(counts, wavelengths)  # trapezoidal area
            particle_counts.append(total_counts)

        particle_counts_mean = np.mean(particle_counts)
        particle_counts_sd = np.std(particle_counts)

        particle_counts_dict[particle_type][frame_num] = {}

        particle_counts_dict[particle_type][frame_num]['mean'] = particle_counts_mean
        particle_counts_dict[particle_type][frame_num]['sd'] = particle_counts_sd


# This is (for now) our final theoretical statistical model for simulations. This includes:
#       - Poisson noise (photons are emitted in a Poisson process)
#       - EM noise (electron multiplication causes more noise)
#       - averaging over 5 lines (done in post-processing)
def generate_spectrum_EMCCD_corrected2(particle_type, frames, G, total_area = None):
    b = 90  # estimated background

    # first get the normalized ground truth
    unif_grid, norm_true_counts_on_grid, list_of_wavelengths, norm_list_of_counts = get_true_spectrum_new_varying_random(particle_type)

    if total_area == None:  # if not given a total area (total brightness), we generate a random one
        # generate a random value for the total counts (basically the brightness of the particle)
        mean_counts, sd_counts = particle_counts_dict[particle_type][frames]['mean'], particle_counts_dict[particle_type][frames]['sd']
        random_total_area = np.random.normal(mean_counts, sd_counts, 1)
        total_area = random_total_area

    # multiply the normalized "ground truth" by the total counts to get a non-normalized "truth"
    mean_spectrum_not_norm = norm_true_counts_on_grid * total_area

    # get a realistic scaled grid (just use the grid of one of the particles) CHANGE LATER !!!! D:LKJFSD:LJ
    realistic_grid = list_of_wavelengths[0]

    # project our "truth" onto the realistic grid
    projected_mean = np.interp(realistic_grid, unif_grid, mean_spectrum_not_norm)

    # now, we want to add random Poisson noise + shot noise to each point (each count)

    list_of_random_frame_avgs = []

    for _ in range(5):  # 5-line averaging correction
        random_frame_avgs = []
        for N in projected_mean:
            # generate a Poisson r.v. with same mean as projected_mean
            this_mean = b + N
            if this_mean < 0:
                this_mean = 0
        
            C_tot_vec = np.random.poisson(this_mean, frames)

            # generate n_frames values of F_i ~ Erling-Gamma(C_tot, 1/G) (these are the EM noise)
            F_vec = np.array([np.random.gamma(shape=k, scale=G) for k in C_tot_vec])

            R_final = (1 / frames) * np.sum(F_vec) / G - b

            random_frame_avgs.append(R_final)
        
        random_frame_avgs = np.array(random_frame_avgs)
        list_of_random_frame_avgs.append(random_frame_avgs)
    
    mean_random_frame_avgs = np.mean(list_of_random_frame_avgs, axis=0)  # 5-line averaging correction

    return realistic_grid, mean_random_frame_avgs, projected_mean, total_area


def generate_many_spectra_EMCCD_corrected2(frames, num_sims_per_particle, other_params):
    G_val = other_params
    sim_spectra = []
    for particle_type in particle_types:
        for i in range(num_sims_per_particle):
            this_sim_wavelengths, this_sim_counts, _, __ = generate_spectrum_EMCCD_corrected2(particle_type, frames, G_val)
            sim_spectra.append([this_sim_wavelengths, this_sim_counts, particle_type])
    return sim_spectra


# generate many spectra of a certain particle type and frame average, formatting it into a 2D numpy array
def generate_many_same_spectra(particle_type, frame_avg, num_sims):
    G_val = 10000
    # first find the length of the wavelength array
    this_sim_wavelengths, this_sim_counts, _, __ = generate_spectrum_EMCCD_corrected2(particle_type, frame_avg, G_val)
    spectra_2d_array = np.zeros((num_sims, len(this_sim_wavelengths)))  # the number of rows is the number of sims. num of columns is length of each spectrum (each wavelength grid)

    for i in range(num_sims):
        this_sim_wavelengths, this_sim_counts, _, __ = generate_spectrum_EMCCD_corrected2(particle_type, frame_avg, G_val)
        spectra_2d_array[i] = this_sim_counts

    return spectra_2d_array  # each row is a sim