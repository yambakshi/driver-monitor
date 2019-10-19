The Driver Monitor is based on OpenVINO's action-recognition demo and is using OpenVINO's driver-action pre-trained models.

To run the driver monitor open cmd and cd to OpenVINO's install folder:
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\

Then, run the script that adds OpenVINO's environment variables:
setupvars.bat

cd to the driver-monitor folder
cd C:\Users\Yam\Documents\Coding Exercises\Python\Guardian\driver-monitor

To start the Driver Monitor, cd to the 'driver_mointor' directory and run the following command:
python monitor_driver.py --encoder models/mobilenetv2/encoder/FP16/driver-action-recognition-adas-0002-encoder.xml --decoder models/mobilenetv2/decoder/FP16/driver-action-recognition-adas-0002-decoder.xml --labels driver_actions_alert.txt

To quit the Driver Monitor press 'q' on the keyboard.
After quitting, the session's video, compressed to VGA resolution and converted to gray-scale, will be saved to the 'driver_monitor' directory with the name: output.avi.