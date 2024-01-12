import os
import subprocess

class ScriptGeneratorModel():
    
    def __init__(self, model_path, predict_script, checkpoint_path):
        self.model_path = model_path
        self.predict_script_path = predict_script
        self.checkpoint_path = checkpoint_path
        
    def predict(self, amrs):
        data_file_path = self.write_dataset_file(amrs)
        script = "sh "
        script += self.model_path
        script += self.predict_script_path
        script += " " + self.checkpoint_path
        script += " " + data_file_path
        script += " 0" 
        os.chdir(self.model_path)
        return run_shell_script(script)

        
        
    def write_dataset_file(self, items):
        file_path = self.model_path + '/dataset.txt'
        file = open(file_path,'w')
        for item in items:
            file.write(item+"\n"+"\n")
        file.close()
        return file_path
        
        
        
def run_shell_script(script):
    try:
        # Use subprocess.run to execute the shell script
        result = subprocess.run(script.split(), check=True, text=True, capture_output=True)

        # Print the output of the shell script
        print("Shell script output:")
        print(result.stdout)

        # Return the exit code of the shell script
        return result.returncode

    except subprocess.CalledProcessError as e:
        # Handle errors if the shell script fails
        print(f"Error running shell script: {e}")
        return e.returncode

