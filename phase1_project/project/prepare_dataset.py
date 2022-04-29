import librosa
import os
import json

PATH_TO_DATASETS = "dataset"
PATH_TO_JSON = "data.json"
SAMPLES_TO_CONSIDER = 22050 

'below function will help us to prepare and dump the datasets in the valid formate and will save in json formate'
def datasets_prepration_preprocessing_steps(path_to_datasets, path_to_json, mfcc_cofficient_number=13, n_point_fft=2048, assigned_hop_length=512):
   
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

   
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path_to_datasets)):

        if dirpath is not path_to_datasets:

           
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            
            for f in filenames:
                file_path = os.path.join(dirpath, f)

              
                signal, sample_rate = librosa.load(file_path)

               
                if len(signal) >= SAMPLES_TO_CONSIDER:

                   
                    signal = signal[:SAMPLES_TO_CONSIDER]

                   
                    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=mfcc_cofficient_number, n_fft=n_point_fft,
                                                 hop_length=assigned_hop_length)

                  
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))

    
    with open(path_to_json, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    datasets_prepration_preprocessing_steps(PATH_TO_DATASETS, PATH_TO_JSON)
