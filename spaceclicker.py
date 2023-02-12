import argparse
import time
from matplotlib import pyplot as plt
from scipy import signal
import csv
import pyautogui
import numpy as np
from numpy import genfromtxt
import utils

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets


def main():
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=True)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.NO_BOARD)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = 'COM3'
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board

    board = BoardShim(args.board_id, params)

    # read the template
    filt = genfromtxt('reading1.csv', delimiter=',')
    filt = filt[:,3574:3700] # 2 channels x 125 samples

    # normalize each channel
    norm_ch1 = (filt[0,:] - np.mean(filt[0,:]))/np.std(filt[0,:])
    norm_ch2 = (filt[1,:] - np.mean(filt[1,:]))/np.std(filt[1,:])


    # now set up the session
    board.prepare_session()
    board.start_stream()
    list1=[]
    
    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:

            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            time.sleep(0.6)
            data = board.get_current_board_data(126)

            # Only keep the channels we're interested in
            data_ch1 = (data[1,:]-np.mean(data[1,:]))/np.std(data[1,:])
            data_ch2 = (data[2,:]-np.mean(data[2,:]))/np.std(data[2,:])

            matches_1 = signal.correlate(norm_ch1,data_ch1)
            matches_2 = signal.correlate(norm_ch2,data_ch2)

            matches_1 = np.abs(matches_1)
            matches_2 = np.abs(matches_2)

            maxMatch_1 = np.max(matches_1)
            maxMatch_2 = np.max(matches_2)
            avgmaxmatch = (maxMatch_1+maxMatch_2)/2

            list1.append(avgmaxmatch)
            if avgmaxmatch > 70:
                pyautogui.press("space")
     
    except KeyboardInterrupt:
        board.stop_stream()
        board.release_session()
        print('Blink')
        print('Closing!')
        #plt.title("avg maxmatches") 
        #plt.xlabel("reading")
        #plt.ylabel("value")
        #plt.plot(list1)
        #plt.show()

if __name__ == "__main__":
    main()