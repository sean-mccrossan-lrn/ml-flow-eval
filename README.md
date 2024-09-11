# LLM Evaluation Repository
This repository is used for experimenting with and evaluating various LLMs (Large Language Models) and prompts for the team. Metrics specific to each use case can be captured and analyzed.

## Prerequisites
* poetry
* Python 3.9


# Poetry Installation on macOS

This guide will help you install and configure Poetry, a Python dependency and packaging manager, on your macOS machine.

## Prerequisites

- macOS with Python installed (preferably Python 3.8+)
- Basic knowledge of terminal usage

## Step 1: Install Homebrew (if not already installed)

Poetry can be installed via the official installer, but Homebrew is often used to manage macOS packages. If you don't have Homebrew installed, you can install it by running the following command in your terminal:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
### Once Homebrew is installed, update it:
```
brew update
```

## Step 2: Install Poetry Using the Official Installer
The easiest way to install Poetry on macOS is via the official installation script. Run the following command to install Poetry:
```
curl -sSL https://install.python-poetry.org | python3 -
```
Note: This will install Poetry in your user's home directory ($HOME/.poetry) by default.

## Step 3: Add Poetry to Your Path
To use Poetry globally from your terminal, add the following line to your shell configuration file (~/.zshrc, ~/.bashrc, or ~/.bash_profile, depending on your shell):

For Zsh (the default shell on macOS since Catalina):
```
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```
For Bash:
```
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
```

# Step 4: Verify the Installation
Once Poetry is installed and your terminal path is updated, you can verify that Poetry is working correctly by running:

```
poetry --version

```
##  Step 5: Creating a New Poetry Project
Now that Poetry is installed, you can create a new Python project using the following command:
```
poetry new my-project
```
## Step 6: Managing Dependencies with Poetry
To add dependencies to your project, navigate to your project directory and use the following command:
```
cd my-project
poetry add <package-name>
```

For example, to add requests:
```
poetry add requests
```

## Step 7: Activate the Virtual Environment
Poetry automatically manages virtual environments for your projects. To activate the virtual environment for your project, run:
```
poetry shell
```
Now, any commands you run will use the dependencies installed in the Poetry-managed virtual environment.

## Step 8: Installing Dependencies from pyproject.toml
If you have an existing project with a pyproject.toml file, you can install all the dependencies by running:
```
poetry install
```

# Setting up .env for projects

**Very often you will need an .env file that could be shared across experiments.**
In case like using GPTs and other models, you will need to have a .env file with the following content:
```bash
OPENAI_API_KEY=<your-openai-api-key>
```
create this file in the root directory  (ai-labbook) and create a symlink to it in the experiment directory for each of the experiments that use it.
For example `language-translation` experiment uses it, so you can create a symlink to it in the `language-translation` directory.
```shell
ln -s <path_to_your_projects_dir>/language-translation/.env <path_to_your_projects_dir>/language-translation/
```
you can run the same command for all the experiments that use the same .env file. Any changes in the `.env` file will be reflected in all the experiments that use it.

# Steps to set up aws environment


To set up an AWS profile in the credentials file on your machine, follow these steps:

## Step 1: Install AWS CLI (if not already installed)
If you don't already have the AWS CLI installed, you can install it using Homebrew:
```
brew install awscli
```
## Step 2: Configure AWS Credentials
You can configure your AWS credentials by running the following command in your terminal:

```
aws configure
```
This will prompt you to enter your AWS Access Key ID, Secret Access Key, default region, and output format.

## Step 3: Manually Add or Edit the AWS Credentials File
If you need to manually set up or edit your AWS credentials file, follow these steps:

- Open the AWS credentials file, typically located at ~/.aws/credentials. If the file does not exist, you can create it manually.

``` 
nano or vim ~/.aws/credentials
```
- Add your AWS profile details. You can have multiple profiles in this file. Here's an example of how you might set up a profile:
```
[profile-name] eg: learnosity-ds-us
aws_access_key_id = YOUR_DEFAULT_ACCESS_KEY
aws_secret_access_key = YOUR_DEFAULT_SECRET_KEY
aws_region = YOUR_REGION

[profile-name]
aws_access_key_id = YOUR_PROFILE_ACCESS_KEY
aws_secret_access_key = YOUR_PROFILE_SECRET_KEY
aws_region = YOUR_REGION
```