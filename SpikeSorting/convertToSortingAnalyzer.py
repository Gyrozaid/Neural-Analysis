# %%
import spikeinterface
from spikeinterface import load_waveforms
# %%
waveformPath = "/Volumes/Rat/Chris_Perk/NAc_Ephys/SortingOutputs/OutputCPWI18_2019-08-01_08-47-01_RLLRRL 420uA/WaveformOutClean"
outPath = "/Volumes/Rat/Chris_Perk/NAc_Ephys/SortingOutputs/OutputCPWI18_2019-08-01_08-47-01_RLLRRL 420uA/SortingAnalyzerOutClean"

extractor = load_waveforms(folder=waveformPath)
analyzer = extractor.sorting_analyzer
analyzer.save_as(folder=outPath,format="binary_folder")
