# cloudiod
Development and deployment of Cloudiod, a containerized web service for real-time image object detection using YOLO and OpenCV, hosted on a Kubernetes cluster.

The Cloudiod project entails the creation of a sophisticated web-based image object detection system. Users can upload images to the Cloudiod web service, which processes these images to detect and identify objects within them. This detection service utilizes the YOLO (You Only Look Once) library, renowned for its real-time object detection capabilities, and OpenCV, a powerful open-source computer vision and machine learning library. Both libraries are implemented using Python.

The project is deployed within a containerized environment using Docker, and Kubernetes is employed for container orchestration. The web service is designed as a RESTful API built with Flask, allowing for concurrent processing of multiple client requests. Images are sent as base64-encoded JSON objects via HTTP POST requests, and the web service returns a JSON response containing the detected objects, their accuracy scores, and bounding box coordinates.

**Key components of the project include:**

**Web Service Development:** A multi-threaded Flask-based RESTful API that handles image uploads, processes them using YOLO and OpenCV, and returns detection results in JSON format.

**Dockerization:** Creating a Docker image for the web service to ensure consistent and isolated execution across different environments.

**Kubernetes Deployment:** Setting up a Kubernetes cluster on Oracle Cloud Infrastructure (OCI) with one controller and two worker nodes. The cluster runs multiple pods of the web service, managed through Kubernetes deployment and service configurations.

**Load Testing and Performance Analysis:** Conducting experiments to evaluate the system's performance under varying loads and resources. This involves testing with different numbers of client threads and Kubernetes pods to measure the impact on response times.

The project includes comprehensive documentation and a detailed report analyzing the performance metrics. Additionally, a video demonstration showcases the setup, configuration, and operation of the Cloudiod system, providing insights into its architecture and implementation.
