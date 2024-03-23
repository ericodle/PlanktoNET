# PlanktoNET
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="https://github.com/ericodle/PlanktoNET/blob/main/img/D20230307T053258_IFCB108_02078.png" alt="Logo" width="600" height="300">
  </a>

<h3 align="center">PlanktoNET: Automated identification of planktonic biodiversity *Research Ongoing* </h3>

  <p align="center">
  This repository houses code generated during development of PlanktoNET, which is intended for a broader project exploring the biodiversity of plankton around Japan. This project is a joint effort between research teams at the Okinawa Institute of Science and Technology and Hokkaido University.
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name/issues">Report Bug</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Request Feature</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About this Project

  <p align="center">
  The primary objective of this project is to develop an integrated "default" sorting mechanism. Upon completion of supervised training, users will have the capability to sort their images without the necessity of manual intervention. Subsequently, the model can be further refined and expanded to accommodate individual datasets and additional locations within the database. This has consistently been the overarching objective of the project.
    <br />
  </p>

  <p align="center">
  <img src="https://github.com/ericodle/PlanktoNET/blob/main/img/workflow.png" alt="Logo" width="700" height="500">
    <br />
  </p>
  
  Experiment Breakdown:
  
  1. Develop a predictive image sorting model utilizing data from McLane Labs.
  2. Evaluate model performance on IFCB and PlanktoScope imaging technologies.
  3. Implement a finetuning feature to allow users to incorporate additional data into their unique image databases.
 </p>
## Prerequisite

Install [Python3](https://www.python.org/downloads/) on your computer.

Enter this into your computer's command line interface (terminal, control panel, etc.) to check the version:

  ```sh
  python --version
  ```

If the first number is not a 3, update to Python3.

## Setup

Here is an easy way to use our GitHub repository.

### Step 1: Clone the repository


Open the command line interface and run:
  ```sh
  git clone git@github.com:ericodle/PlanktoNET.git
  ```

You have now downloaded the entire project, including all its sub-directories (folders) and files.
(We will avoid using Git commands.)

### Step 2: Navigate to the project directory
Find where your computer saved the project, then enter:

  ```sh
  cd /path/to/project/directory
  ```

If performed correctly, your command line interface should resemble

```
user@user:~/PlanktoNET$
```

### Step 3: Create a virtual environment: 
Use a **virtual environment** so library versions on your computer match the versions used during development and testing.


```sh
python3 -m venv -p python3.8 planktonet-env
```

A virtual environment named "planktonet-env" has been created. 
Enter the environment to do our work by using the following command:


```sh
source planktonet-env/bin/activate
```

When performed correctly, your command line interface prompt should look like 

```
(planktonet-env) user@user:~/PlanktoNET$
```

### Step 3: Install requirements.txt

Avoid "dependency hell" by installing specific software versions known to work well together.

  ```sh
pip install -r requirements.txt
  ```

### Step 4: How to use PlanktoNET

PlanktoNET appeals to the broader plankton research community by offering a modular set of utilities.

Specificaly, users can...

#### Sort images using a new model ####
The training scripts require 4 arguments: 1) train/test image directory path, 2) output directory, 3) initial learning rate 4) number of images per class.
For example, if you want to train your own Vision Transformer model, run:

```sh
python3 ./src/experiment_1/train_vision_transformer.py ./mclanelabs/mclanelabs_set ./experiment_1/transformer 0.0001 300
```

#### Sort images using an existing model ####

#### Fine-tune an existing model on new data ####

#### Evaluate a model on curated ground truth data ####


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
