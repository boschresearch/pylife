import hudson.tasks.junit.TestResultSummary

// SonarQube related variable - FEEL FREE TO MODIFY
BASE_BRANCH_NAME = "develop"

// SonarQube related variables
SONAR_PROPERTY_FILE = "sonar-project.properties"
CURRENT_BRANCH_NAME = env.BRANCH_NAME
PR_KEY = env.CHANGE_ID
SONARQUBE_SERVER_ID = "SonarqubeMSO"

@Library('create-jenkins-library') _
pipeline {
    // Which Build Node?
    agent any
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
                bat 'install_pylife_win.bat'
            }
        }
        // Test Python packages with PyTest
        stage('PyTest') {
            steps {
                // Running unit tests
                bat 'batch_scripts/run_pylife_tests.bat'
            }
        }
        // Static code analysis with Flake8
        stage('Flake8') {
            steps {
                catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                    bat 'batch_scripts/run_code_analysis.bat'
                }
            }
        }
        // Building documentation
        stage('Documentation') {
            steps {
                catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                    bat 'batch_scripts/build_docs.bat'
                }
            }
        }
        stage ('Publish Test Results') {
            steps {
                // JUnit Test results
                junit 'junit.xml'

                //Publish
                publishHTML target: [
                    allowMissing: false,
                    alwaysLinkToLastBuild: false,
                    keepAll: true,
                    reportDir: 'coverage_report',
                    reportFiles: 'index.html',
                    reportName: 'Test coverage'
                ]
            }
        }
        stage ('Publish coverage report') {
            steps{
                script {
                    cobertura(
                        coberturaReportFile: "coverage_report.xml",
                        onlyStable: false,
                        failNoReports: true,
                        failUnhealthy: false,
                        failUnstable: false,
                        autoUpdateHealth: true,
                        autoUpdateStability: true,
                        zoomCoverageChart: true,
                        maxNumberOfBuilds: 0,
                        lineCoverageTargets: '75, 75, 75',
                        conditionalCoverageTargets: '75, 75, 75',
                        classCoverageTargets: '75, 75, 75',
                        methodCoverageTargets: '75, 75, 75',
                        fileCoverageTargets: '75, 75, 75',
                    )
                }
            }
        }
        stage ('Publish documentation') {
            steps{
                publishHTML target: [
                    allowMissing: false,
                    alwaysLinkToLastBuild: false,
                    keepAll: true,
                    reportDir: 'docs/_build/',
                    reportFiles: 'index.html',
                    reportName: 'Documentation'
                ]
            }
        }
		stage ('SonarQube analysis') {
			steps {
				sonarScanner (
					sonarqubeServerID: SONARQUBE_SERVER_ID, 
					currentBranchName: CURRENT_BRANCH_NAME,
					sonarBaseBranchName: BASE_BRANCH_NAME,
					prKey: PR_KEY,
					workSpacePath: env.WORKSPACE
				)
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
                    [$class: 'RequesterRecipientProvider']
                ]
            }
        }
    }
}