pipeline {
    // Which Build Node?
    agent {
        label 'master'
    }
    // Default python project settings (EDITABLE)
    environment {
        VENV_NAME = '_pyLife'
    }    
    // Discard the old builds and artifacts
    options {
      buildDiscarder logRotator(
          artifactDaysToKeepStr: '30', 
          artifactNumToKeepStr: '1', 
          daysToKeepStr: '30', 
          numToKeepStr: '10'
        )
    }
    // Build stages
    stages {
        // Create a new Anaconda python virtual environment and set proxy variables
        stage('Prepare Python env') {
            steps {
                bat 'set PATH'
                bat 'conda env remove -n %VENV_NAME%'
                bat 'conda create -y -n %VENV_NAME% --file ./requirements_CONDA.txt'
            }
        }
        // Install python packages with pip to the already created python environment
        stage('Pip install') {
            steps {
                // Using proxy user credentials
                withCredentials([usernameColonPassword(credentialsId: 'TECH_USER', variable: 'PASS')]) {
                    bat 'activate %VENV_NAME%'
                    bat 'pip install -r requirements_PIP.txt --proxy http://%PASS%@rb-proxy-de.bosch.com:8080 --user'
                }
            }
        }        
    }
}