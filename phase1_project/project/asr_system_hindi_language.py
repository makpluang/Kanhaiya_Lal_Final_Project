import librosa
import tensorflow as tf
import numpy as np

DESIGNED_MODEL_PATH = "model.h5"
OVERALL_SAMPLE_TO_BE_TAKEN = 22050

class _Asr_System_Hindi_Language:

    model = None
    _mapping = [
        "chota",
        "yah",
        "tum",
        "namaskar",
        "dekhna",
        "mai",
        "bada",
        "apne",
        "upar",
        "adheek",
        "yha",
        "vha",
        "niche",
        "lekin",
        "vah",
        "dayen",
        "kuch",
        "bayein",
        "hum",
        "kya",
        "hamara"
    ]
    _instance = None


    def predict(self, file_path):

        MFCCs = self.preprocess(file_path)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):

        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= OVERALL_SAMPLE_TO_BE_TAKEN:

            signal = signal[:OVERALL_SAMPLE_TO_BE_TAKEN]

            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T


def Asr_System_Hindi_Language():
    if _Asr_System_Hindi_Language._instance is None:
        _Asr_System_Hindi_Language._instance = _Asr_System_Hindi_Language()
        _Asr_System_Hindi_Language.model = tf.keras.models.load_model(DESIGNED_MODEL_PATH)
    return _Asr_System_Hindi_Language._instance




if __name__ == "__main__":

    kss = Asr_System_Hindi_Language()
    kss1 = Asr_System_Hindi_Language()
    assert kss is kss1

  
    keyword = kss.predict("test/kya.mp3")
    print(keyword)
