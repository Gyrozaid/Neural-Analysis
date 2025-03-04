
def main():
    import spikeinterface.full as si
    import spikeinterface.exporters as se
    from probeinterface import generate_tetrode, ProbeGroup, generate_linear_probe
    import warnings
    import argparse
    import os
    import torch
    warnings.simplefilter("ignore")
    print(f"SpikeInterface version: {si.__version__}")

    parser = argparse.ArgumentParser(
        description="arg parser for data paths"
    )
    
    parser.add_argument(
        '-p', '--path',
        type=str,
        required=True,
        help='enter path to your data'
    )
    
    parser.add_argument(
        '-s', '--save_path',
        type=str,
        required=True,
        help='enter path to your saved results'
    )
    
    parser.add_argument(
        '-a', '--sorter_alg',
        type=str,
        required=True,
        help='enter name of sorting algorithm'
    )

    parser.add_argument(
        '-l', '--recording_length',
        type=str,
        required=False,
        help='enter length of recording if you are slicing it. Used for testing'
    )
    
    parser.add_argument(
        '-g', '--grouping',
        type=int,
        required=False,
        help='enter how many probe groups are in the recording. Ensure groupings are divisible by # electrodes'
    )

    
    args = parser.parse_args()

    path_to_data = args.path
    path_to_results = args.save_path
    algorithm = args.sorter_alg
    length = args.recording_length
    
    #read in data
    if len(os.listdir(path_to_data + "\\experiment1\\recording1\\continuous")) == 2:
        full_raw_rec = si.read_openephys(path_to_data, stream_id="1")
    else:
        full_raw_rec = si.read_openephys(path_to_data)
        
    #select non aux channels
    channel_ids = full_raw_rec.get_channel_ids()
    full_raw_rec = full_raw_rec.select_channels([channel_id for channel_id in channel_ids if channel_id[0] != 'A'])



    #make the probe group
    if args.grouping == None:
        channel_names = list(full_raw_rec.get_channel_ids())
        probe_group = ProbeGroup()

        for i in range(int(len(channel_names) / 4)):
            tetrode = generate_tetrode()
            tetrode.move([i * 100, 0])
            tetrode.set_contact_ids([4*i, (4*i)+1, (4*i)+2, (4*i)+3])
            probe_group.add_probe(tetrode)
            
    else:
        channel_names = list(full_raw_rec.get_channel_ids())
        probe_group = ProbeGroup()

        for i in range(int(len(channel_names) / args.grouping)):
            linear_probe = generate_linear_probe(num_elec=args.grouping)
            linear_probe.move([i * 100, 0])
            contact_ids = []
            for j in range(args.grouping):
                contact_ids.append(args.grouping*i + j)
                
            linear_probe.set_contact_ids(contact_ids)
            probe_group.add_probe(linear_probe)

        
    #sets decive channel indices (just 0 through # channels) to map to the channel names
    #therefore, electrodes 0-3 are tetrode 0, 4-7 are tetrode 1, ect based on device channel indices, NOT channel names, which are irrelevent
    #in phy, electrodes will be labeled with device channel index not channel name
    
    probe_group.set_global_device_channel_indices([i for i in range(len(channel_names))])
    raw_rec = full_raw_rec.set_probegroup(probe_group, group_mode="by_probe")
    
    #slice recording for testing
    if length:
        try:
            length = int(length)  
            if length > 0:
                recording_loaded = raw_rec.frame_slice(start_frame=0, end_frame=length * 30000)
                print(f"First {length} seconds taken for sorting")
            else:
                raise ValueError("Length must be a positive integer.")
        except ValueError as e:
            print(f"Invalid recording length: {e}")
            return
    else:
        recording_loaded = raw_rec
        
    #run kilosort
    if algorithm == 'kilosort4':
        nearest_chans = 4 if args.grouping == None else args.grouping
        whitening_range = 4 if args.grouping == None else args.grouping
        
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        
        print(f"CUDA available: {cuda_available}. Using device: {device}")

        sorted_recording = si.run_sorter("kilosort4", 
                                    recording_loaded, 
                                    folder=path_to_results, 
                                    remove_existing_folder = True, 
                                    verbose=True,       
                                    nblocks=0,
                                    nearest_chans=nearest_chans,
                                    whitening_range=whitening_range,
                                    save_preprocessed_copy=True,
                                    torch_device=device
                                    )
        
    #run mountainsort
    elif algorithm == 'mountainsort5':
        #filter and whiten
        filtered = si.bandpass_filter(recording_loaded, freq_min=300, freq_max=6000)
        whitened = si.whiten(filtered, dtype='float32')
        recording_loaded = whitened
        
        
        #sort
        sorted_recording = si.run_sorter_by_property(
            sorter_name = "mountainsort5",
            recording = recording_loaded,
            grouping_property = 'group',
            scheme = '3',
            folder=path_to_results + f'\\mountainsort5_sorting'
        )
        
        #compute necessary metrics
        sorting_analyzer = si.create_sorting_analyzer(sorted_recording, 
                                            recording_loaded, 
                                            folder=path_to_results + "\\mountainsort5_sorting_analyzer",
                                            format="binary_folder",
                                            sparse=True, 
                                            overwrite=True)
        
        sorting_analyzer.compute(["random_spikes", "waveforms", 'correlograms', 'spike_amplitudes', 
                        'templates','unit_locations', 'template_similarity', 'noise_levels', 
                        'isi_histograms', 'principal_components', 'quality_metrics', 'spike_locations', 'template_metrics'])

        
        #export to phy
        se.export_to_phy(sorting_analyzer, path_to_results + "\\mountainsort5_phy")
        
    #run spykingcircus2
    elif algorithm == 'spykingcircus2':
        #filter and whiten
        filtered = si.bandpass_filter(recording_loaded, freq_min=300, freq_max=6000)
        whitened = si.whiten(filtered, dtype='float32')
        recording_loaded = whitened
        
        #run sorter
        sorted_recording = si.run_sorter("spykingcircus2", 
                                        recording_loaded, 
                                        folder=path_to_results + f'\\spykingcircus2_sorting', 
                                        remove_existing_folder = True, 
                                        verbose=True,
                                        clustering = {'legacy': False},
                                        apply_preprocessing=False,
                                        detection= {"peak_sign": "neg", "detect_threshold": 4},
                                        cache_preprocessing= {"mode": "memory", "memory_limit": 0.1, "delete_cache": True},
                                        general= {'ms_before': 2, 'ms_after': 2, 'radius_um': 50}
                                        
                                        )

        #compute necessary metrics
        sorting_analyzer = si.create_sorting_analyzer(sorted_recording, 
                                            recording_loaded, 
                                            folder=path_to_results + "\\spykingcircus2_sorting_analyzer",
                                            format="binary_folder",
                                            sparse=False, 
                                            overwrite=True)
        
        sorting_analyzer.compute(["random_spikes", "waveforms", 'correlograms', 'spike_amplitudes', 
                        'templates','unit_locations', 'template_similarity', 'noise_levels', 
                        'isi_histograms', 'principal_components', 'quality_metrics', 'spike_locations', 'template_metrics'])

        #export to phy
        se.export_to_phy(sorting_analyzer, path_to_results + "\\spykingcircus2_phy")
        


if __name__ == "__main__":
    main()