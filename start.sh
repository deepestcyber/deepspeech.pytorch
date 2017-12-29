#!/bin/sh

sudo chown -R deepcyber /dev/snd/
pulseaudio -D
pacmd set-default-source 'alsa_input.usb-OmniVision_Technologies__Inc._USB_Camera-B4.09.24.1-01.multichannel-input'
