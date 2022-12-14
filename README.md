Image processing methods divided into two branches, face detection methods and face recognition methods.

Here we practice a face detection method which find and count faces in an image. The main contribution can be foud in the following link.
https://github.com/deepinsight/insightface

But here, we try to have a web-app using streamlit package to facilitate face detection process.

The following Figure show a sample of face detection process.

![pp1](https://user-images.githubusercontent.com/84702784/203147362-5ec5beb2-123d-4d7e-9d81-d679e1a68146.png)

As you see, it counts faces and determines the pixel number for each face bounding box. Enlarge the image to see faces bounding box.

[http://ehsan-picture.ir/](http://ehsan-picture.ir/) is a url let you have an online evaluation. Upload your image and enjoy.

-------------------------------------------------------------------------------------------------------------------

The following command run the python code.

streamlit run vision.py

It walks all images in the folder named IMG and detect human face. Or else, you may upload your desired image to see the results.

