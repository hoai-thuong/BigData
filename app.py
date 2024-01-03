from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

import pandas as pd
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, flash, redirect, url_for, request
from flask_mongoengine import MongoEngine #ModuleNotFoundError: No module named 'flask_mongoengine' = (venv) C:\flaskmyproject>pip install flask-mongoengine  
from werkzeug.utils import secure_filename
import os
#import magic
import urllib.request
  
from flask import Flask
# Creating a Flask Instance
app = Flask(__name__)

from flask_mongoengine import MongoEngine #ModuleNotFoundError: No module named 'flask_mongoengine' = (venv) C:\flaskmyproject>pip install flask-mongoengine  
app.config['MONGODB_SETTINGS'] = {
    'db': 'final',
    'host': 'localhost',
    'port': 27017
}
db = MongoEngine()
db.init_app(app)   



IMAGE_SIZE = (128, 128)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Loading Pre-trained Model ...")
model = load_model('./model/bidl.h5')

# def image_preprocessor(path):
#     '''
#     Function to pre-process the image before feeding to the model.
#     '''
#     print('Processing Image ...')
#     currImg_BGR = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     print("Original Image Shape:", currImg_BGR.shape)
#     currImg_resized = cv2.resize(currImg_BGR, IMAGE_SIZE)
#     currImg_resized = currImg_resized / 255.0
#     currImg = np.reshape(currImg_resized, (1, 128, 128, 1))
#     print("Resized Image Shape:", currImg.shape)
#     return currImg

def image_preprocessor(path):
    '''
    Function to pre-process the image before feeding to the model.
    '''
    print('Processing Image ...')
    currImg_BGR = cv2.imread(path)
    if currImg_BGR is None:
        print(f"Unable to read image at {path}")
        return None  # Return None if image reading fails
    
    b, g, r = cv2.split(currImg_BGR)
    currImg_RGB = cv2.merge([r, g, b])
    
    # Resize the image to match the model's input size
    currImg_resized = cv2.resize(currImg_RGB, (64, 64))
    if currImg_resized is None:
        print(f"Unable to resize image at {path}")
        return None  # Return None if resizing fails
    
    currImg_resized = currImg_resized / 255.0
    currImg_resized = np.reshape(currImg_resized, (1, 64, 64, 3))
    
    return currImg_resized

def model_pred(image):
    '''
    Performs predictions based on input image
    '''
    print("Image_shape", image.shape)
    print("Image_dimension", image.ndim)
    
    prediction_prob = model.predict(image)[0]
    print("Raw Probabilities:", prediction_prob)

    # Compare the probabilities of the two classes
    probability_normal = prediction_prob[0]
    probability_pneumonia = prediction_prob[1]
    
    print("Probability of Normal:", probability_normal)
    print("Probability of Pneumonia:", probability_pneumonia)

    if probability_normal > probability_pneumonia :
        return "Normal"
    else:
        return "Pneumonia"



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
  
class bacsi(db.Document):
    name = db.StringField()
    age = db.StringField()
    region = db.StringField()
    profile_pic = db.StringField()
    result = db.StringField()

     

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    # Checks if post request was submitted
    if request.method == 'POST':
        file = request.files['imageFile']
        # print(request.form)
        print("Request Method:", request.method)
        print("Request Form Data:", request.form)
        print("Request Files:", request.files)
        rs_username = request.form['name']
        inputEmail = request.form['age']
        inputRegion = request.form['region']

        filename = secure_filename(file.filename)

        '''
        - request.url - http://127.0.0.1:5000/
        - request.files - Dictionaary of HTML elem "name" attribute and corrospondiong file details eg. 
        "imageFile" : <FileStorage: 'Profile_Pic.jpg' ('image/jpeg')>
        '''
        # check if the post request has the file part
        if 'imageFile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        # check if filename is an empty string
        file = request.files['imageFile']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # if file is uploaded
        if file and allowed_file(file.filename):
            imgPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(imgPath)
            print(f"Image saved at {imgPath}")
            # Preprocessing Image
            image = image_preprocessor(imgPath)
            # Perfroming Prediction
            pred = model_pred(image)
            usersave = bacsi(name=rs_username, age=inputEmail, region=inputRegion, profile_pic=file.filename, result = pred)
            usersave.save()
            return render_template('upload.html', name=filename, result=pred)
    return redirect(url_for('home'))



@app.route('/statistics')
def statistics():
    # Fetch all fields from MongoDB
    data = bacsi.objects.only('name', 'age', 'region', 'profile_pic', 'result')
    data_list = list(data)

    # Convert each document to a dictionary and exclude "_id"
    data_list_no_id = [{k: v for k, v in entry.to_mongo().items() if k != '_id'} for entry in data_list]

    # Create a DataFrame from the data
    df = pd.DataFrame(data_list_no_id)

    # Check if expected columns are present
    expected_columns = ['name', 'age', 'region', 'profile_pic', 'result']
    if not all(column in df.columns for column in expected_columns):
        return f"Error: Missing expected columns in DataFrame. Expected columns: {expected_columns}"

    # Select only the required columns
    df = df[['name', 'age', 'region', 'profile_pic', 'result']]

    # Count the occurrences of each region and result
    counts = df.groupby(['region', 'result']).size().reset_index(name='count')

    # Pivot the data to have regions as index, results as columns, and counts as values
    pivot_df = counts.pivot(index='region', columns='result', values='count').fillna(0)

    # Normalize the counts to percentages
    # pivot_df['Pneumonia_percentage'] = pivot_df['Pneumonia'] / (pivot_df['Normal'] + pivot_df['Pneumonia']) * 100
    # pivot_df['Normal_percentage'] = pivot_df['Normal'] / (pivot_df['Normal'] + pivot_df['Pneumonia']) * 100
    pivot_df['Pneumonia_percentage'] = pivot_df['Pneumonia'] / (pivot_df['Normal'] + pivot_df['Pneumonia']) * 100
    pivot_df['Normal_percentage'] = pivot_df['Normal'] / (pivot_df['Normal'] + pivot_df['Pneumonia']) * 100

    # x Get the top 5 provinces with the largest number of pneumonia cases
    top_provinces = pivot_df.sort_values(by='Pneumonia', ascending=False).head(5)

# Plot the grouped bar chart for regions using Matplotlib
    region_plot_buffer = generate_plot(top_provinces, 'Top 5 Provinces with Largest Pneumonia Cases')

    # Plot the grouped bar chart for regions using Matplotlib
    # region_plot_buffer = generate_plot(pivot_df, 'Pneumonia Cases by Region')

    # Count the occurrences of each result
    result_counts = df['result'].value_counts()
    result_counts = result_counts.astype(float)  # Convert to float

    # Plot the pie chart
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue'])
    ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Percentage of Pneumonia and Normal Cases')

    # Save the figure to a BytesIO buffer
    buffer_pie = BytesIO()
    plt.savefig(buffer_pie, format='png')
    buffer_pie.seek(0)
    image_png_pie = buffer_pie.getvalue()
    buffer_pie.close()

    # Convert the pie chart image to a base64-encoded string
    image_base64_pie = base64.b64encode(image_png_pie).decode()

    # Embed the pie chart image in the HTML using a data URI
    pie_chart_html = f'<img src="data:image/png;base64,{image_base64_pie}">'

    # Fetch age data from MongoDB
    data_age = bacsi.objects()

    # Convert each document to a dictionary and exclude "_id"
    data_list_age = [{k: v for k, v in entry.to_mongo().items() if k != '_id'} for entry in data_age]

    # Create a DataFrame from the age data
    df_age = pd.DataFrame(data_list_age)

    # Check if expected columns are present
    expected_columns_age = ['name', 'age', 'region', 'profile_pic', 'result']
    if not all(column in df_age.columns for column in expected_columns_age):
        return f"Error: Missing expected columns in DataFrame. Expected columns: {expected_columns_age}"

    # Select only the required columns
    df_age = df_age[['name', 'age', 'region', 'profile_pic', 'result']]

    # Convert 'age' column to integer type
    df_age['age'] = df_age['age'].astype(int)

    # Define age groups
    bins = [0, 18, 29, 39, 49, 100]
    labels = ['<18', '18-29', '30-39', '40-49', '>50']

    # Create 'age_group' column based on defined age groups
    df_age['age_group'] = pd.cut(df_age['age'], bins=bins, labels=labels, right=False)

    # Count the occurrences of each age group and result
    counts_age = df_age.groupby(['age_group', 'result']).size().reset_index(name='count')

    # Pivot the age data to have age groups as index, results as columns, and counts as values
    pivot_df_age = counts_age.pivot(index='age_group', columns='result', values='count').fillna(0)

    # Normalize the counts to percentages
    pivot_df_age['Pneumonia_percentage'] = pivot_df_age['Pneumonia'] / (pivot_df_age['Normal'] + pivot_df_age['Pneumonia']) * 100
    pivot_df_age['Normal_percentage'] = pivot_df_age['Normal'] / (pivot_df_age['Normal'] + pivot_df_age['Pneumonia']) * 100

    # Plot the grouped bar chart for age using Matplotlib
    age_plot_buffer = generate_plot(pivot_df_age, 'Pneumonia Cases by Age Group')

    # Convert the images to base64-encoded strings
    region_plot_base64 = base64.b64encode(region_plot_buffer.getvalue()).decode()
    age_plot_base64 = base64.b64encode(age_plot_buffer.getvalue()).decode()

    # Embed the images in the HTML using data URIs
    region_plot_html = f'<img src="data:image/png;base64,{region_plot_base64}">'
    age_plot_html = f'<img src="data:image/png;base64,{age_plot_base64}">'

    return render_template('statistics.html', region_plot_html=region_plot_html, age_plot_html=age_plot_html, pie_chart_html=pie_chart_html)

def generate_plot(pivot_df, title):
    # Plot the grouped bar chart using Matplotlib
    fig, ax = plt.subplots()
    width = 0.35
    separation = 0
    categories = range(len(pivot_df))
    pneumonia_percentage = pivot_df['Pneumonia_percentage']
    normal_percentage = pivot_df['Normal_percentage']

    # Plot the bars for "Pneumonia" and "Normal" side-by-side with increased separation
    ax.bar(categories, pneumonia_percentage, width, label='Pneumonia')
    ax.bar([c + width + separation for c in categories], normal_percentage, width, label='Normal')

   
    # Explicitly set the ticks and labels on the x-axis
    labels = pivot_df.index
    ax.set_xticks(categories)
    ax.set_xticklabels(labels, rotation=10)


    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.legend()

    # Save the figure to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer

if __name__ == '__main__':
    app.run(debug=True)
