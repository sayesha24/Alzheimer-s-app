import pickle
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score
import cv2
from kivy.core.window import Window
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

model_path = '/Users/sayeshagoel/Desktop/inspirit ai /my_trained_model2.h5'
model_path_LR = 'LR_model.pickle'

if os.path.exists(model_path):
    classifier = load_model(model_path)
else:
    raise FileNotFoundError(f"The model file was not found at {model_path}")

class AlzheimerPredictor:
    def __init__(self):
        self.model = self.load_model()
    
    def load_model(self):
        try:
            with open(model_path_LR, 'rb') as file:
                model = pickle.load(file)
            return model
        except FileNotFoundError:
            return self.train_model()
    
    def train_model(self):
        AD = pd.read_csv('stats.csv')
        AD['new_column'] = AD['Diagnostic'].apply(lambda x: 1 if x == "Alzheimer's Disease" else 0)
        biomarkers = ['CSF Amyloid (pg/mL)', 'CSF Total tau (pg/mL)', 'CSF Phosphorylated tau (pg/mL)']
        X = AD[biomarkers].to_numpy()
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        y = AD['new_column']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train logistic regression model
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save the trained model
        with open(model_path_LR, 'wb') as file:
            pickle.dump(lr, file)
        
        return lr
    
    def predict(self, csf_amyloid, csf_total_tau, csf_phosphorylated):
        try:
            new_data = np.array([[float(csf_amyloid), float(csf_total_tau), float(csf_phosphorylated)]])
        except ValueError:
            return "Please enter valid numeric values for all inputs."
        
        prediction = self.model.predict(new_data)
        
        if prediction[0] == 1:
            return "Prediction: Alzheimer's Disease"
        else:
            return "Prediction: No Alzheimer's Disease"

class MainApp(App):
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)
        self.layout = FloatLayout()
        self.img = None
        self.solution = TextInput(text="ALZHEIMER'S DISEASE DETECTION", halign="center", padding=5, font_size=50, background_color=(0.7, 0.3, 0.8), size_hint=(1, 0.1), pos_hint={'center_x': 0.5, 'center_y': 0.95})
        self.layout.add_widget(self.solution)

        self.show_error_popup("Please use this only if the MMSE score was less than 26.")

        MRI_button = Button(
                    text="Submit Brain MRI scan",
                    size_hint=(None, None), size=(800, 50),
                    pos_hint={'center_x': 0.5, 'center_y': 0.4}
                    )
        MRI_button.bind(on_press=self.mri)
        self.layout.add_widget(MRI_button)

        biomarkers_button = Button(
                    text="Submit Biomarkers",
                    
                    size_hint=(None, None), size=(800, 50),
                    pos_hint={'center_x': 0.5, 'center_y': 0.6}
                    )
        biomarkers_button.bind(on_press=self.biomarkers)
        self.layout.add_widget(biomarkers_button)

        return self.layout
    
    def biomarkers(self, instance):
        
        self.layout.clear_widgets()
        self.solution = TextInput(text="ALZHEIMER'S DISEASE DETECTION", halign="center", padding=5, font_size=50, background_color=(0.7, 0.3, 0.8), size_hint=(1, 0.1), pos_hint={'center_x': 0.5, 'center_y': 0.95})
        self.layout.add_widget(self.solution)

        self.csf_amy = TextInput(
                        hint_text="Add CSF amyloid value",
                        size=(800, 50), size_hint=(None, None),
                        foreground_color=(1, 0, 0, 1),
                        pos_hint={'center_x': 0.5, 'center_y': 0.6}
                    )
        self.layout.add_widget(self.csf_amy)
                    
        self.csf_total = TextInput(
                        hint_text="Add CSF Total Tau value",
                        size_hint=(None, None), size=(800, 50),
                        pos_hint={'center_x': 0.5, 'center_y': 0.5}
                    )
        self.layout.add_widget(self.csf_total)
                    
        self.csf_phospho = TextInput(
                        hint_text="Add CSF Phosphorylated Tau value",
                        size_hint=(None, None), size=(800, 50),
                        pos_hint={'center_x': 0.5, 'center_y': 0.4}
                    )
        self.layout.add_widget(self.csf_phospho)
                
        predict_button = Button(
                        text="Predict",
                        size_hint=(None, None), size=(800, 50),
                        pos_hint={'center_x': 0.5, 'center_y': 0.3}
                        )
        predict_button.bind(on_press=self.predict)
        self.layout.add_widget(predict_button)

        MRI_button = Button(
                    text="Submit Brain MRI scan",
                    size_hint=(None, None), size=(800, 50),
                    pos_hint={'center_x': 0.5, 'center_y': 0.1}
                    )
        MRI_button.bind(on_press=self.mri)
        self.layout.add_widget(MRI_button)


        

    def predict(self, instance):
        try:
            csf_amyloid = float(self.csf_amy.text)
            csf_total_tau = float(self.csf_total.text)
            csf_phosphorylated = float(self.csf_phospho.text)
        except ValueError:
            self.show_error_popup("Please enter valid numeric values for all inputs.")
            return
        
        predictor = AlzheimerPredictor()
        result_text = predictor.predict(csf_amyloid, csf_total_tau, csf_phosphorylated)
        self.show_popup(result_text)

    def show_popup(self, result_text):
        # Create a popup layout
        popup_layout = BoxLayout(orientation='vertical')
        
        # Create a label with the result text
        result_label = Label(text=result_text)
        popup_layout.add_widget(result_label)
        
        # Create a button to close the popup
        close_button = Button(text='Close')
        close_button.bind(on_press=lambda x: popup.dismiss())
        popup_layout.add_widget(close_button)
        
        # Create the popup
        popup = Popup(title='Prediction Result',
                      content=popup_layout,
                      size_hint=(None, None), size=(400, 200))
        
        # Open the popup
        popup.open()

    def mri(self, instance):
        self.layout.clear_widgets()
        self.solution = TextInput(text="ALZHEIMER'S DISEASE DETECTION", halign="center", padding=5, font_size=50, background_color=(0.7, 0.3, 0.8), size_hint=(1, 0.1), pos_hint={'center_x': 0.5, 'center_y': 0.95})
        self.layout.add_widget(self.solution)
        self.upload_photo = Button(
            text="Upload Photo",
            background_color=(0, 0, 1, 1),
            size_hint=(None, None),
            size=(300, 50),
            pos_hint={'center_x': 0.5, 'center_y': 0.6}
        )
        self.upload_photo.bind(on_press=self.open_file_chooser)
        self.layout.add_widget(self.upload_photo)

    def open_file_chooser(self, instance):
        content = BoxLayout(orientation='vertical')
        self.filechooser = FileChooserIconView()
        submit_button = Button(text="Submit", size_hint_y=None, height=50)
        submit_button.bind(on_press=self.submit_selection)

        content.add_widget(self.filechooser)
        content.add_widget(submit_button)

        self.filechooser_popup = Popup(
            title="Select Image",
            content=content,
            size_hint=(0.9, 0.9)
        )
        self.filechooser_popup.open()

        self.image = Image(size_hint=(0.8, 0.8), pos_hint={'center_x': 0.5, 'center_y': 0.3})
        self.layout.add_widget(self.image)

    def submit_selection(self, instance):
        selection = self.filechooser.selection
        if selection:
            self.image.source = selection[0]
            self.img = cv2.imread(selection[0])
            self.filechooser_popup.dismiss()

            self.find_result = Button(text="Make Prediction", background_color=(0, 0, 1, 1),
                                      size_hint=(None, None),
                                      size=(300, 50),
                                      pos_hint={'center_x': 0.5, 'center_y': 0.18})
            self.find_result.bind(on_press=self.make_prediction)
            self.layout.add_widget(self.find_result)

    def make_prediction(self, instance):
        # Remove previous prediction label if it exists
             # Remove previous prediction label if it exists
        if hasattr(self, 'pred_label') and self.pred_label:
                            self.layout.remove_widget(self.pred_label)
                        
                        # Remove the "Upload Photo" button if it exists
        if hasattr(self, 'select_photo') and self.select_photo:
                            self.layout.remove_widget(self.select_photo)

        if self.img is not None:
             try:
                        # Preprocess the image for the model
                                    img_resized = cv2.resize(self.img, (128, 128))
                                    img_array = np.array(img_resized, dtype=np.float32)
                                
                                
                            # Normalize the image if necessary (assuming normalization was done during training)
                                    img_array /= 255.0
                                    
                                    img_array = np.expand_dims(img_array, axis=0)

                                    print(f"Image array shape: {img_array.shape}")  # Debugging info

                            # Make prediction
                                    prediction = classifier.predict(img_array)
                                    print(f"Prediction a -rray: {prediction}")  # Debugging info

                                    result = np.argmax(prediction, axis=1)[0]

                                    # Map the prediction to the corresponding class label
                                    class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
                                    result_text = class_labels[result]

                            # Display the result in the solution TextInput
                                    self.pred_label = Label(text=f'Prediction: {result_text}', size_hint=(None, None), color="black", size=(400, 50),
                                                    pos_hint={'center_x': 0.5, 'center_y': 0.8})
                                    self.layout.add_widget(self.pred_label)
                    
                    # Add the button to allow uploading a new image
                                    self.new_upload_button = Button(text="Upload New Image", background_color=(0, 0, 1, 1),
                                                                size_hint=(None, None),
                                                                size=(300, 50),
                                                                pos_hint={'center_x': 0.5, 'center_y': 0.1})
                                    self.new_upload_button.bind(on_press=self.open_file_chooser)
                                    self.layout.add_widget(self.new_upload_button)
                            
             except Exception as e:
                                            print(f"Error during prediction: {e}")
                                            self.solution.text = f"Prediction failed: {e}"
             else:
                                        self.solution.text = 'No image selected or captured.'           
                


    

    def show_error_popup(self, error_message):
        # Create a popup layout
        popup_layout = BoxLayout(orientation='horizontal')
        
        # Create a label with the error message
        error_label = Label(text=error_message)
        popup_layout.add_widget(error_label)
        
        # Create the popup
        popup = Popup(title='NOTICE',
                      content=popup_layout,
                      size_hint=(None, None), size=(800, 200))
        
        # Open the popup
        popup.open()

if __name__ == "__main__":
    MainApp().run()
