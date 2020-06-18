import sys
import os
import shutil
import xml.etree.ElementTree as ET

coverage_home_folder = sys.argv[1]
print(coverage_home_folder)
coverage_backup_path = f'{coverage_home_folder}coverage_report_backup.xml'
print(coverage_backup_path)
coverage_old_xml_path = f'{coverage_home_folder}coverage_report.xml'
print(coverage_old_xml_path)
coverage_new_xml_path = 'coverage_report.xml'
print(coverage_new_xml_path)

current_folder = os.getcwd() 
print("Current Directory", current_folder) 

is_develop = "develop" in current_folder
print(is_develop)

coverage_old_xml = ET.parse(coverage_old_xml_path)
coverage_new_xml = ET.parse(coverage_new_xml_path)

coverage_new = coverage_new_xml.getroot()
coverage_old = coverage_old_xml.getroot()

if coverage_new.attrib['line-rate'] < coverage_old.attrib['line-rate']:
    raise Exception("Test coverage is decreasing! You have to write unit tests to cover your new code lines")
else:
    print("Test coverage is sufficient")
    if is_develop:
        try:
            os.remove(coverage_backup_path)
            shutil.move(coverage_old_xml_path, coverage_backup_path)
            shutil.copyfile(coverage_new_xml_path, coverage_old_xml_path)    
        except IOError:
            print("One of the coverage files is not accessible")