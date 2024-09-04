import yaml
import json
import os
import dota_utils as util

def json2yaml():
    filenames = util.GetFileFromThisRootDir('/Users/deeptisaravanan/Downloads/valdotaAnnotation/')
    
    for file in filenames:
        if('.DS_Store' not in file):
            with open(file, 'r') as f:
                configuration = json.load(f)
            destfile = file
            if('.json' in destfile):
                destfile = destfile.replace('.json', '.yml')
                destfile = destfile.replace('valdotaAnnotation', 'valdotayml')
            with open(destfile, 'w') as f_out:
                yaml.dump(configuration, f_out)
            #with open(destfile, 'r') as yaml_file:
                #print(yaml_file.read())        

if __name__ == '__main__':
    json2yaml()