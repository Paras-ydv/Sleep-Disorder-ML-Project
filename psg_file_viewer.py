import mne
import matplotlib.pyplot as plt
psg_file = 'D:/Sleep_EDF_data/SC4001E0-PSG.edf'

# Load the PSG data
raw_data = mne.io.read_raw_edf(psg_file, preload=True)

# Display basic information about the PSG file
print(raw_data.info)

# Plot the data to visualize the EEG channels
raw_data.plot(duration=30, n_channels=10)  # Show 10 channels over 30 seconds
plt.show()