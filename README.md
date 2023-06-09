# J_PlanktoNET
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="https://github.com/ericodle/J_PlanktoNET/blob/main/D20230307T053258_IFCB108_02078.png" alt="Logo" width="600" height="300">
  </a>

<h3 align="center">J_PlanktoNET: Automated identification of planktonic biodiversity around Okinawa and Hokkaido, Japan</h3>

  <p align="center">
  This repository houses code generated during development of J_PlanktoNET for a broader project exploring the biodiversity of plankton around Japan. The project is a joint effort between X/Y/Z with the Okinawa Institute of Science and Technology and Eric Odle/Dr. Kevin Wakeman with Hokkaido University.
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name/issues">Report Bug</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Request Feature</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About this Project

Automated data collection is enabled using an imaging flowcytobot (FCB) which
generates large datasets that can be used for monitoring marine biodiversity.
However, the bottleneck for analysis is often that species identification requires
specific expertise and is extremely time consuming.
Currently, the available software packages do not have their training dataset and
code available which restricts the flexibility for application by groups or locations.
Here we propose to develop an open-source software package designed for batch
identification of plankton using a classifier to process data collected from an FCB.
Samples will be collected from ecologically different locations, Okinawa and
Hokkaido; 3 sites in Okinawa (Sesoko island, Ikei Island, and Seragaki) and 2 sites in
Hokkaido (Otaru, Akkeshi). A training dataset will be generated from samples
collected which we will use for training and testing.

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting Started

Download this repository by going up to the green "Code" button at the top right and clicking "Download ZIP".

Alternatively, you can also clone the repo directly using the following command.

  ```sh
  # Replace "your_folderpath_here" with the actual folder where you want the project to go.
  cd /your_folderpath_here
  git clone git@github.com:ericodle/J_PlanktoNET.git
  ```

> __For this example, the working directory is the repository root directory.__ 

### Install dependencies using pip

  ```sh
  # Install dependencies if necessary. 
  # You may want to work in a virtual environment. Conda environments are nice for that.
  pip install torch torchvision
  ```

### Download test dataset

> The training/testing image dataset containing around 7,000 photos can be obtained (here). These images were acquired during a single collection session using the Imaging FlowCytobot (https://mclanelabs.com/imaging-flowcytobot). 
> We recommend downloading the images directly into the root project folder.

```sh
# This script will do something.
./script.py
```

### Train a model from scratch

Run the following script to define, instantiate, and train our custom Encoder/Decoder models used alongside pre-trained VGG16.
   ```sh
   # Set up a new training run
   ./train_model.py
   ```
   
Note #1: Training requires a GPU to complete in a timely manner. You can either use your own hardware, or work on a Colab environment.
If you use a GPU, make sure you have cuda and all related dependencies set up in your environment.

Note #2: Training is as much an art as it is a science, and often involves playing around with different hyperparameters. Users are encouraged to go into the train_model.py script and change the optimizer, learning rate, epochs, or other parameters. The default settings represent what worked best for us at the time of experimentation.

### Testing a trained model

You now have a model trained from scratch on MFCCs extracted from the GTZAN music genre dataset. Nice! It is time to see how well it can classify musical genre.
In our conference paper, we used a shuffled 80:10:10 split for training, train phase validation, and testing. Therefore, the music clip segments reserved for testing come from same dataset, but have never been seen by the trained model before. Given the scope of the GTZAN dataset, your trained model is unlikely to distinguish Bunun polyphonic chant music from Ainu rimse dance music. A neural network is only as good as the data on which it is trained. Within the GTZAN training data, how well can your model classify musical genre?

  ```sh
  # Test a pre-trained model.
  ./test_model.py
  ```




<!-- CONTRIBUTING -->
## Contributing

Contributions make the open source community great. Everyone has a unique combination of skills and experience. Your input is **highly valued**.
If you have ideas for improvement, please fork the repo and create a pull request. 
If this is your first pull request, just follow the steps below:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the GNU Lesser General Public License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


Citing
------

Please cite the future paper, coming soon!



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
