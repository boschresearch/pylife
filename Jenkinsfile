pipeline {
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
                bat 'conda create -n %VENV_NAME%' --file ./requirements_CONDA.txt
            }
        }
    }
}