from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
import numpy as np
import pickle, timeit, random
from sklearn.metrics import f1_score, accuracy_score
from Preprocessing import convertToGrayscale, align_image, load_image
from Feature_extractor import embed_image
from logger import logger

log = logger('Classification')

class Classifier:

    def __init__(self, method='svc'):
        self.method = method
        self.model_path = "./saved/svc_prob.pkl"
        if method == 'knn': self.model_path = './saved/knn_prob.pkl'
        elif method == 'rfc': self.model_path = './saved/rfc_prob.pkl'
        self.encoder_path = "./saved/encoder.pkl"
        self.validate_every = 2

    def prepareData(self, data):

        embedded = np.zeros((data.shape[0], 128))
        exclude_indices = []
        for i, item in enumerate(data):
            img = load_image(item.image_path())
            embedding = embed_image(img)

            if type(embedding) is not np.ndarray:
                exclude_indices.append(i)
            else:
                embedded[i] = embedding

        cleaned_data = np.delete(data, exclude_indices)
        embedded = np.delete(embedded, exclude_indices, axis=0)

        return cleaned_data, embedded


    def trainClassifier(self, data):
        start = timeit.default_timer()

        data, embedded = self.prepareData(data)

        targets = np.array([m.name for m in data])

        encoder = LabelEncoder()
        encoder.fit(targets)

        pickle.dump(encoder, open(self.encoder_path, 'wb'))

        # Numerical encoding of identities
        y = encoder.transform(targets)

        train_idx = np.arange(data.shape[0]) % self.validate_every != 0
        test_idx = np.arange(data.shape[0]) % self.validate_every == 0

        X_train = embedded[train_idx]
        X_test = embedded[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        target_test = targets[test_idx]
        if self.method == 'knn':
            model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        elif self.method == 'rfc':
            model =RandomForestClassifier(n_estimators=100, oob_score = True ,n_jobs = 1,random_state =1)
        else:
            model = SVC(kernel="linear", probability=True)

        model.fit(X_train, y_train)

        pickle.dump(model, open(self.model_path, 'wb'))

        end = timeit.default_timer()
        log.info("Classifying took: %s" % (end - start))

        predictions = model.predict(X_test)
        #predict_probabilities=model.predict_proba(X_test)

        acc = accuracy_score(y_test, predictions)

        print('Classification accuracy = %s' % acc)
        return self


    def classify(self, img):

        img_embedding = embed_image(img)

        with open(self.model_path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)

        predicted = model.predict_proba([img_embedding])[0]

        print(predicted)

        predicted_index = np.argmax(predicted)
        probability =predicted[predicted_index]

        print(predicted_index)

        #if predicted[predicted_index] < 0.3: return 'unknown'#(for test_8)
        if predicted[predicted_index] < 0.4: return 'unknown'#(for test_7, test_2,test_5,test,)


        with open(self.encoder_path, 'rb') as file:
            encoder = pickle.load(file)

        img_class = encoder.inverse_transform(predicted_index)

        #return str(img_class)
        return str(img_class).split("-")[-1]
        #return str(img_class).split("-")[-1], probability