from __future__ import print_function
from openvino.inference_engine import IECore
from action_recognition.result_renderer import ResultRenderer
from action_recognition.models import IEModel
from action_recognition.steps import run_pipeline
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Put down the phone exercise.')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    parser.add_argument("--encoder", help="Required. Path to encoder model", required=True, type=str)
    parser.add_argument("--decoder", help="Required. Path to decoder model", required=True, type=str)
    parser.add_argument("--cpu_extension",
                      help="Optional. For CPU custom layers, if any. Absolute path to a shared library with the "
                           "kernels implementation.", type=str, default=None)
    parser.add_argument("--device",
                      help="Optional. Specify a target device to infer on. CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for the device specified. "
                           "Default value is CPU",
                      default="CPU", type=str)
    parser.add_argument("--fps", help="Optional. FPS for renderer", default=30, type=int)
    parser.add_argument("--labels", help="Optional. Path to file with label names", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    camera_device = args.camera

    if args.labels:
        with open(args.labels) as f:
            labels = [l.strip() for l in f.read().strip().split('\n')]
    else:
        labels = None

    result_presenter = ResultRenderer(labels=labels)
    ie = IECore()
    if 'MYRIAD' in args.device:
        myriad_config = {"VPU_HW_STAGES_OPTIMIZATION": "YES"}
        ie.set_config(myriad_config, "MYRIAD")

    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    decoder_target_device = "CPU"
    if args.device != 'CPU':
        encoder_target_device = args.device
    else:
        encoder_target_device = decoder_target_device

    encoder_xml = args.encoder
    encoder_bin = encoder_xml.replace(".xml", ".bin")
    decoder_xml = args.decoder
    decoder_bin = decoder_xml.replace(".xml", ".bin")
    encoder = IEModel(encoder_xml, encoder_bin, ie, encoder_target_device, num_requests=(3 if args.device == 'MYRIAD' else 1))
    decoder = IEModel(decoder_xml, decoder_bin, ie, decoder_target_device, num_requests=2)
    run_pipeline(camera_device, encoder, decoder, result_presenter.render_frame, fps=30)


if __name__ == "__main__":
    main()
