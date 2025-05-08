from flask import Flask, render_template, request, session, flash, send_file

import mysql.connector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'aaa'


@app.route('/')
def home():
    return render_template('index.html')






@app.route("/Predict")
def Predict():
    return render_template('Predict.html')


@app.route("/preidct", methods=['GET', 'POST'])
def preidct():
    if request.method == 'POST':
        import os
        file = request.files['file']
        fname = file.filename
        file.save('static/Out/test.jpg')

        import warnings
        warnings.filterwarnings('ignore')
        import tensorflow as tf
        classifierLoad = tf.keras.models.load_model('model.h5')
        import numpy as np
        from keras.preprocessing import image
        base_dir = 'Dataset/train/'
        catgo = os.listdir(base_dir)
        img = image.load_img('static/Out/test.jpg', target_size=(200, 200))
        org = 'static/Out/test.jpg'
        test_image = np.expand_dims(img, axis=0)
        result = classifierLoad.predict(test_image)
        ind = np.argmax(result)

        print(ind)
        print(catgo[ind])

        out = ''
        pre = ''
        predicted_class = catgo[ind]

        out = predicted_class

        if (predicted_class == "esophagitis"):
            out = predicted_class

            pre = 'Fundoplication: A surgery that strengthens the lower esophageal sphincter to prevent acid from ' \
                  'coming back up into the esophagus '

        elif predicted_class == "normal":
            out = predicted_class
            pre = 'Nil '


        elif (predicted_class == "polyps"):
            out = predicted_class
            pre = 'Medications like steroids and cortisone can help stop nasal polyps from growing. Surgery options ' \
                  'include polypectomy, balloon sinuplasty, and functional endoscopic sinus surgery (FESS). '

        elif (predicted_class == "ulcerativecolitis"):
            out = predicted_class
            pre = 'Anti-inflammatory drugs: The most common treatment, these can be taken orally or topically. ' \
                  'Examples include aminosalicylates, such as mesalamine, balsalazide, and olsalazine. '

    return render_template('Result.html', org=org, result=pre, pre=out)


@app.route("/preidct2", methods=['GET', 'POST'])
def preidct2():
    if request.method == 'POST':
        return render_template('Predict.html')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
