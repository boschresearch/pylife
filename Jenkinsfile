pipeline {
    // Which Build Node?
    agent {
        label 'master'
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
        // Create a new Anaconda python virtual environment and install PIP packages
        stage('Prepare Python env') {
            steps {
                bat 'install_pylife.bat'
            }
        }
        // Test Python packages with PyTest
        stage('PyTest & Code coverage') {
            steps {
                bat 'batch_scripts/run_pylife_tests.bat'
            }
        }
        // Static code analysis with Flake8
        stage('Flake8') {
            steps {
                bat 'batch_scripts/run_code_analysis.bat'
            }
        }
        // Publish Test results with MSTest
        stage('Publish Test Results') {
            steps {
                // JUnit Test results
                junit 'junit.xml'

                // Test Coverage results
                publishCoverage adapters: [coberturaAdapter(mergeToOneReport: true, path: 'coverage_report.xml')], failNoReports: true, sourceFileResolver: sourceFiles('NEVER_STORE')
            }
        }
        // Clean up the Python virtual environment
        stage('Clean env') {
            steps {
                bat 'conda env remove -p ./_venv'
            }
        }                        
    }
    // Post-build actions
    post {
        always {
            // Sending emails to developers & Culprits & Requester
            script {
                emailext body: '''${SCRIPT, template="groovy-html.template"}''',
                mimeType: 'text/html',
                subject: "[Jenkins] ${currentBuild.result}: '${env.JOB_NAME} [Build #${env.BUILD_NUMBER}]'",
                recipientProviders: [
                    [$class: 'CulpritsRecipientProvider'], 
                    [$class: 'DevelopersRecipientProvider'], 
                    [$class: 'RequesterRecipientProvider']
                ]
            }
        }
    }    
}