import marimo

__generated_with = "0.7.20"
app = marimo.App(width="medium")


@app.cell
def __():
    import shutil 
    import pathlib
    from pathlib import Path
    import os

    #pathToData = "/mnt/Ephys/NAc_Ephys_raw_data/USB_Copy_2022-12-09_120822"
    pathToData = "/Volumes/Lab/Ephys/RawData"

    #Determine each animal folder path and extract the actual name for the animals folder
    recFolders = {}
    for file in os.listdir(path=pathToData):
        fileStr = str(file)
        if fileStr[:3].lower() == "cpw":
            fileLoc = Path(f"{pathToData}/{file}")
            recFolders[fileStr] = str(fileLoc)
        #elif fileStr[:8].lower() != "cpwi17_s":
        #    for file2 in os.listdir(path=f"{pathToData}/{file}"):
        #        fileLoc2 = Path(f"{pathToData}/{file}/{file2}")
        #        folders.append(fileLoc2)
        #        folderNames.append(str(file2))
    print(recFolders)

    import spikeinterface.full as si
    global_job_kwargs = dict(n_jobs=-1, chunk_duration="1s",progress_bar=True)
    si.set_global_job_kwargs(**global_job_kwargs)
    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as pp


    from probeinterface import Probe, ProbeGroup, generate_tetrode
    from probeinterface.plotting import plot_probe_group
    import numpy as np
    import pandas as pd


    from spikeinterface import sorters

    from spikeinterface.postprocessing import compute_spike_amplitudes, compute_correlograms, compute_template_similarity
    from spikeinterface.qualitymetrics import compute_quality_metrics, compute_num_spikes
    from spikeinterface.exporters import export_report, export_to_phy
    import spikeinterface.qualitymetrics as qual
    from spikeinterface import create_sorting_analyzer

    from spikeinterface.postprocessing import compute_principal_components

    import spikeinterface.postprocessing as spost

    from datetime import date
    from datetime import datetime

    import pickle

    import math

    #Create log file to keep track of erros in pipeline
    import logging
    now = datetime.now()
    logString = now.strftime("%d%m%Y %H%M%S")
    logging.basicConfig(filename=f"logs/{logString}run.log", force=True)
    process = True

    sortingObjects = []
    #Iterate over every animal folder selected in first cell
    for recordingName, recordingLoc in recFolders.items():
        pipelineBroke = False
        probeMismatch = False
        correctChannels = True
        recName = recordingName
        #Change these as desired
        finalDir = Path(f"/Volumes/Lab/Ephys/SortingOutputs/Output{recName}")
        directory = Path(f"/Volumes/Lab/Ephys/EphysOutputs/Output{recName}")
        dataSource = recordingLoc
        print(f"Data Source: {dataSource}")
        #dataCopyLoc = Path(f"/home/moormanlab/Documents/CPWI17/{recName}")
        numberOfOrigChannels = 0
        
        #Don't sort if already sorted
        #if (not Path.is_dir(dataCopyLoc)) and (not Path.is_dir(finalDir)):
        if True:
            #print("Copying")
            #shutil.copytree(
            #    src=dataSource,
            #    dst=dataCopyLoc
            #    )
            #print("Done Copying")
            totalExperiments = len(next(os.walk(dataSource))[1])
            blockIndex = int(totalExperiments) - 1
            try:
                dataLocal = se.read_openephys(folder_path=dataSource, stream_id = '1', block_index = blockIndex)
            except Exception as e:
                print(e)
                print("Tried 1")
                try:
                    dataLocal = se.read_openephys(folder_path=dataSource, stream_id = '0', block_index = blockIndex)
                except Exception as e:
                    print(e)
                    print("Tried 0")
                    continue
            #Generic bandpass filter
            recordingObject = pp.bandpass_filter(dataLocal, freq_min=300, freq_max=6000)
            numberOfOrigChannels = int(recordingObject.get_num_channels())
            if (numberOfOrigChannels == 64) or (numberOfOrigChannels == 67):
                correctChannels = True
                #channel_ids = np.array(channel_ids)
                print(recordingObject)
                '''
                try:
                    numChannels = 67
                    x = [0] * numChannels
                    y = list(range(1, numChannels+1))
                    locations = list(zip(x,y))
                    locations = np.array(locations)
                    locations.shape
                    origChanIDs = recordingObject.get_channel_ids().tolist()
                    auxChannels = ['AUX1','AUX2','AUX3','aux1','aux2','aux3']
                    auxChannels2 = ['AUX1','AUX2','AUX3']
                    newChanIDs = [channel for channel in origChanIDs if channel not in auxChannels]
                    newChanIDs = np.asarray(newChanIDs)
                    recordingObject.set_channel_locations(locations,origChanIDs)
                    recordingObject = recordingObject.remove_channels(auxChannels2)
                '''
                #Remove AUX channels
                origChanIDs = recordingObject.get_channel_ids().tolist()
                process = True
                auxChannels = ['AUX1','AUX2','AUX3','aux1','aux2','aux3']
                newChanIDs = [channel for channel in origChanIDs if channel not in auxChannels]
                
                #Old code block to remove excess channels for odd numbers
                numChannels = len(newChanIDs)
                totalTetrodeChannels = int(4 * math.floor(numChannels/4))
                channelsToCut = newChanIDs[:numChannels-totalTetrodeChannels]
                #newChanIDs = newChanIDs[:-channelsToCut or None]
                
                #Create pseudo channel locations
                x = [0] * numChannels
                y = list(range(1, numChannels+1))
                locations = list(zip(x,y))
                locations = np.array(locations)
                locations.shape
                newChanIDs = np.asarray(newChanIDs)
                recordingObject.set_channel_locations(locations,newChanIDs)
                #badChannelIDs, labels = si.detect_bad_channels(recording)
                recordingObject = recordingObject.remove_channels(channelsToCut)
                print(recordingObject)
                print(recordingObject.get_channel_ids())
                import re
                channel_nums = recordingObject.get_channel_ids()
                num_channels = recordingObject.get_num_channels()
                channel_nums = channel_nums[0:totalTetrodeChannels].tolist()
                for i in range(len(channel_nums)):
                    word = str(channel_nums[i])
                    word = re.findall(r'\d+' , word)
                    channel_nums[i] = int(word[0]) - 1

                #Generate tetrodes on a "probe"
                probeGroup = ProbeGroup()
                for i in range(int(totalTetrodeChannels/4)):
                    tetrode = generate_tetrode()
                    tetrode.move([i * 50, 0])
                    probeGroup.add_probe(tetrode)
                probeGroup.set_global_device_channel_indices(channel_nums)
                #probeGroup.set_global_device_channel_indices(np.arange(totalTetrodeChannels))
                df = probeGroup.to_dataframe()
                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', None)
                print(df)
                #plot_probe_group(probeGroup, with_channel_index=True, same_axes=True)
                #print(channel_nums)
                #recording = pp.phase_shift(recording)
                try:
                    recordingWProbe = recordingObject.set_probegroup(probeGroup,group_mode="by_probe")
                    recordingCom = pp.common_reference(recordingWProbe, operator='median', reference="global")
                    #print(recordingCom)
                    print("Channel Groups: " + str(recordingCom.get_channel_groups()))
                    print("Channel IDs: " + str(recordingCom.get_channel_ids()))
                    print("Channel Locations: " + str(recordingCom.get_channel_locations()))
                except Exception as e:
                    logging.error(f"{e}")
                    probeMismatch = True

            elif (numberOfOrigChannels == 32):
            #elif (numberOfOrigChannels == 32) or (numberOfOrigChannels == 35):
                correctChannels = True
                import numpy as np
                #channel_ids = np.array(channel_ids)
                print(recordingObject)
                '''
                try:
                    numChannels = 67
                    x = [0] * numChannels
                    y = list(range(1, numChannels+1))
                    locations = list(zip(x,y))
                    locations = np.array(locations)
                    locations.shape
                    origChanIDs = recordingObject.get_channel_ids().tolist()
                    auxChannels = ['AUX1','AUX2','AUX3','aux1','aux2','aux3']
                    auxChannels2 = ['AUX1','AUX2','AUX3']
                    newChanIDs = [channel for channel in origChanIDs if channel not in auxChannels]
                    newChanIDs = np.asarray(newChanIDs)
                    recordingObject.set_channel_locations(locations,origChanIDs)
                    recordingObject = recordingObject.remove_channels(auxChannels2)
                '''
                origChanIDs = recordingObject.get_channel_ids().tolist()
                process = True
                auxChannels = ['AUX1','AUX2','AUX3','aux1','aux2','aux3']
                newChanIDs = [channel for channel in origChanIDs if channel not in auxChannels]
                numChannels = len(newChanIDs)
                totalTetrodeChannels = int(4 * math.floor(numChannels/4))
                channelsToCut = newChanIDs[:numChannels-totalTetrodeChannels]
                #newChanIDs = newChanIDs[:-channelsToCut or None]
                x = [0] * numChannels
                y = list(range(1, numChannels+1))
                locations = list(zip(x,y))
                locations = np.array(locations)
                locations.shape
                print(locations)
                newChanIDs = np.asarray(newChanIDs)
                recordingObject.set_channel_locations(locations,newChanIDs)
                #badChannelIDs, labels = si.detect_bad_channels(recording)
                recordingObject = recordingObject.remove_channels(channelsToCut)
                print(recordingObject)
                print(recordingObject.get_channel_ids())
                import re
                channel_nums = recordingObject.get_channel_ids()
                num_channels = recordingObject.get_num_channels()
                channel_nums = channel_nums[0:totalTetrodeChannels].tolist()
                for i in range(len(channel_nums)):
                    word = str(channel_nums[i])
                    word = re.findall(r'\d+' , word)
                    channel_nums[i] = int(word[0]) - 1

                probeGroup = ProbeGroup()
                for i in range(int(totalTetrodeChannels/4)):
                    tetrode = generate_tetrode()
                    tetrode.move([i * 50, 0])
                    probeGroup.add_probe(tetrode)
                probeGroup.set_global_device_channel_indices(channel_nums)
                #probeGroup.set_global_device_channel_indices(np.arange(totalTetrodeChannels))
                df = probeGroup.to_dataframe()
                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', None)
                print(df)
                #plot_probe_group(probeGroup, with_channel_index=True, same_axes=True)
                #print(channel_nums)
                #recording = pp.phase_shift(recording)
                try:
                    recordingWProbe = recordingObject.set_probegroup(probeGroup,group_mode="by_probe")
                    recordingCom = pp.common_reference(recordingWProbe, operator='median', reference="global")
                    #print(recordingCom)
                    print("Channel Groups: " + str(recordingCom.get_channel_groups()))
                    print("Channel IDs: " + str(recordingCom.get_channel_ids()))
                    print("Channel Locations: " + str(recordingCom.get_channel_locations()))
                except Exception as e:
                    logging.error(f"{e}")
                    probeMismatch = True
            else:
                correctChannels = False
                logging.info(f"Sorting Failed {recName} or Already Exists. Channels:{numberOfOrigChannels}")
                print(f"Sorting Failed {recName} or Already Exists")
                print(f"Number of channels is {numberOfOrigChannels}")
                #shutil.rmtree(dataCopyLoc)
        else:
            continue
        if (correctChannels) and (not probeMismatch):
            #thresholdList = [5.5]
            #Sorts by group (each group is one tetrode)
            for i in range(1):
                sortingWorked = True
                todayDate = date.today()
                fullDir = Path(f"{directory}")
                if not Path.is_dir(fullDir):
                    sortingMS5 = sorters.run_sorter_by_property(
                        sorter_name ="mountainsort5", 
                        recording = recordingCom, 
                        grouping_property ='group',
                        folder =  f"{fullDir}/SortMS5",
                        #sorter parameters
                        scheme = '3',
                        #detect_threshold = thresholdList[i]
                        #detect_sign = -1,
                        #detect_time_radius_msec =  0.5,
                        #snippet_T1 = 20,
                        #snippet_T2 = 20,
                        #npca_per_channel = 3,
                        #npca_per_subdivision = 10,
                        #snippet_mask_radius = 250,
                        #scheme1_detect_channel_radius = 150,
                        #scheme2_phase1_detect_channel_radius = 200,
                        #scheme2_detect_channel_radius = 50,
                        #scheme2_max_num_snippets_per_training_batch = 200,
                        #scheme2_training_duration_sec = 300,
                        #scheme2_training_recording_sampling_mode = 'uniform',
                        #scheme3_block_duration_sec = 1800,
                        #freq_min = 300,
                        #freq_max = 6000,
                        #filter = True,
                        #whiten = True
                    )
                    toAddMS5 = ['MS5',sortingMS5]
                    sortingObjects.append(toAddMS5)
                    sortingT2 = sorters.run_sorter_by_property(
                        sorter_name="tridesclous2",
                        recording = recordingCom,
                        grouping_property = 'group',
                        folder=f"{fullDir}/SortT2"
                    )
                    toaddT2 = ['T2',sortingT2]
                    sortingObjects.append(toaddT2)
                    if sortingWorked:
                        with open(f"{fullDir}/recording.pickle" , 'wb') as f:
                            pickle.dump(recordingCom,f)
                        with open(f"{fullDir}/sortingMS5.pickle" , 'wb') as f:
                            pickle.dump(sortingMS5,f)
                        with open(f"{fullDir}/sortingSKC2.pickle" , 'wb') as f:
                            pickle.dump(sortingT2,f)
                        for sortingObj in sortingObjects:
                            fullDir = f"{fullDir}/{sortingObj[0]}"
                            try:
                                cleaned = True
                                #File path here used for GUI scripts
                                #analyzerLoc = "/home/moormanlab/Documents/OutputMS5Test/QualityMetricsTestFolder"
                                analyzerLoc = f"{fullDir}/WaveformOut"
                                analyzerLocClean = f"{fullDir}/WaveformOutClean"
                                analyzer = create_sorting_analyzer(recordingCom,sortingObj,analyzerLoc,overwrite=True)
                                amplitudes = spost.compute_spike_amplitudes(analyzer)
                                amplitudes0 = amplitudes[0]
                                thresh = -200
                                mask = amplitudes0 >= thresh
                                spikes = analyzer.sorting.to_spike_vector()
                                largeAmpSpikes = spikes[mask]
                                sortingLargeAmp = si.NumpySorting(
                                    spikes=largeAmpSpikes,
                                    sampling_frequency=analyzer.sampling_frequency,
                                    unit_ids=analyzer.unit_ids
                                    )
                                with open(f"{fullDir}/sortingThresh.pickle" , 'wb') as f:
                                    pickle.dump(sortingLargeAmp,f)
                                cleanedAnalyzer = create_sorting_analyzer(recordingCom,sortingLargeAmp,analyzerLocClean,overwrite=True)
                                compute_principal_components(cleanedAnalyzer)
                                numSpikes = compute_num_spikes(cleanedAnalyzer)
                                qualityMetrics = compute_quality_metrics(
                                                    waveform_extractor=cleanedAnalyzer,
                                                    metric_names=['num_spikes','firing_rate','snr','amplitude_cutoff','synchrony','firing_range','drift'],
                                                    qm_params=qual.get_default_qm_params()
                                                )
                                compute_spike_amplitudes(cleanedAnalyzer)
                                compute_correlograms(cleanedAnalyzer)
                                compute_template_similarity(cleanedAnalyzer)
                                print(numSpikes)
                                print(qualityMetrics)
                            except:
                                try:
                                    cleaned = False
                                    logging.error(f"Cleaning Recording {recName} failed. Exporting Original Recording")
                                    analyzerLoc = f"{fullDir}/WaveformOut"
                                    analyzerLocClean = f"{fullDir}/WaveformOutClean"
                                    analyzerExtractor = create_sorting_analyzer(recordingCom,sortingObj,analyzerLoc,overwrite=True)
                                    compute_principal_components(analyzerExtractor)
                                    numSpikes = compute_num_spikes(analyzerExtractor)
                                    qualityMetrics = compute_quality_metrics(
                                                        waveform_extractor=analyzerExtractor,
                                                        metric_names=['num_spikes','firing_rate','snr','amplitude_cutoff','synchrony','firing_range','drift'],
                                                        qm_params=qual.get_default_qm_params()
                                                    )
                                    compute_spike_amplitudes(analyzerExtractor)
                                    compute_correlograms(analyzerExtractor)
                                    compute_template_similarity(analyzerExtractor)
                                    print(numSpikes)
                                    print(qualityMetrics)
                                except Exception as e:
                                    pipelineBroke = True
                                    logging.error(e)
                            if pipelineBroke:
                                logging.error(f"Pipeline broke for {recordingName}")
                            elif cleaned:
                                try:
                                    export_report(cleanedAnalyzer,output_folder= f"{fullDir}/GUIReport",format="pdf")
                                except:
                                    logging.error(f"Failed to export report to {fullDir}")
                                    print(f"Failed to export report to {fullDir}")
                                #try:
                                #    export_to_phy(cleanedWe,output_folder= f"{fullDir}/PhyReport")
                                #except:
                                #    logging.error(f"Failed to export phy to {fullDir}")
                                #    print(f"Failed to export phy to {fullDir}")
                                export_to_phy(cleanedAnalyzer,output_folder= f"{fullDir}/PhyReport")
                            elif not cleaned:
                                try:
                                    export_report(analyzerExtractor,output_folder= f"{fullDir}/GUIReport",format="pdf")
                                except:
                                    logging.error(f"Failed to export report to {fullDir}")
                                    print(f"Failed to export report to {fullDir}")
                                #try:
                                #    export_to_phy(analyzerExtractor,output_folder= f"{fullDir}/PhyReport")
                                #except:
                                #    logging.error(f"Failed to export phy to {fullDir}")
                                #    print(f"Failed to export phy to {fullDir}")
                                export_to_phy(analyzerExtractor,output_folder= f"{fullDir}/PhyReport")
                            #print("Deleting Data Locally")
                            #shutil.rmtree(dataCopyLoc)
                            #print("Done Deleting")
                            #print("Moving Sorting to NAS")
                            if not pipelineBroke:
                                try:
                                    shutil.move(
                                        src=Path(f"{fullDir}"),
                                        dst=Path("/mnt/Ephys/SortingOutputs/")
                                    )
                                except:
                                    logging.error("Output Directory Likely Exists")
                                    print("Directory Likely Exists")

            else:
                logging.error(f"Sorting Failed {fullDir} or Already Exists")
                print(f"Sorting Failed {fullDir} or Already Exists")
                continue
    return (
        Path,
        Probe,
        ProbeGroup,
        amplitudes,
        amplitudes0,
        analyzer,
        analyzerExtractor,
        analyzerLoc,
        analyzerLocClean,
        auxChannels,
        blockIndex,
        channel_nums,
        channelsToCut,
        cleaned,
        cleanedAnalyzer,
        compute_correlograms,
        compute_num_spikes,
        compute_principal_components,
        compute_quality_metrics,
        compute_spike_amplitudes,
        compute_template_similarity,
        correctChannels,
        create_sorting_analyzer,
        dataLocal,
        dataSource,
        date,
        datetime,
        df,
        directory,
        export_report,
        export_to_phy,
        f,
        file,
        fileLoc,
        fileStr,
        finalDir,
        fullDir,
        generate_tetrode,
        global_job_kwargs,
        i,
        largeAmpSpikes,
        locations,
        logString,
        logging,
        mask,
        math,
        newChanIDs,
        now,
        np,
        numChannels,
        numSpikes,
        num_channels,
        numberOfOrigChannels,
        origChanIDs,
        os,
        pathToData,
        pathlib,
        pd,
        pickle,
        pipelineBroke,
        plot_probe_group,
        pp,
        probeGroup,
        probeMismatch,
        process,
        qual,
        qualityMetrics,
        re,
        recFolders,
        recName,
        recordingCom,
        recordingLoc,
        recordingName,
        recordingObject,
        recordingWProbe,
        se,
        shutil,
        si,
        sorters,
        sortingLargeAmp,
        sortingMS5,
        sortingObj,
        sortingObjects,
        sortingT2,
        sortingWorked,
        spikes,
        spost,
        tetrode,
        thresh,
        toAddMS5,
        toaddT2,
        todayDate,
        totalExperiments,
        totalTetrodeChannels,
        word,
        x,
        y,
    )


if __name__ == "__main__":
    app.run()
