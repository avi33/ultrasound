'''
%
%       
%       This function performs DAS beamforming for the RF data reveiced 
%       from a plane wave transmission without beamsteering (0 degrees).
%
%       For each pixel (xj,zj), distances are calculated by summing the 
%       distance from the transmiter to the point and distance from point 
%       to the receiving element as follows:
%           total distance = TX delay path + RX delay path
%           TX_delay = zj
%           RX_delay = sqrt(zj^2 + (xj-x)^2)
%
%
%       "RF_data128" contains received signal from each channel
%       "z_start" is the imaging depth starting point
%       "z_stop" is the imaging depth ending point
%       "image_width" is the required imaging width
%       "delta_x" is the lateral step size between each lines of the image
%       "pitch" is the distance between the centres of adjacent elements
%       "c" is the sound speed
%       "fs" is the sampling frequency
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
import torch
import torch.nn as nn

class DelayAndSumBF(nn.Module):
    def __init__(self, fs, output_sz, dx, dy, z_start, z_stop):
        super().__init__()
        c = 1540 #m/s
        self.dt = 1/fs
        self.dd = c * self.dt
        self.dz = self.dd / 2
        self.x_ax = torch.arange(-output_sz.shape[0] // 2, output_sz.shape[0], dx)
        self.y_ax = torch.arange(-output_sz.shape[1] // 2, output_sz.shape[1], dy)
        self.z_az = torch.arange(z_start, z_stop, self.dz)
    
    def forward(self, x):
        return x

function [Beamformed_DATA, z_axis, x_axis] = Basic_Beamformer(RF_data128, z_start, z_stop, image_width, delta_x, pitch, c, fs)
tic
% Receive data parameters
delta_t = 1/fs;                     % Sampling interval (time)
delta_d = c * delta_t;              % Sampling interval (distance)
receive_depth = size(RF_data128,1);    % Receive samples (depth)
N_elements = size(RF_data128,2);       % Number of receive channels = Number of elements
% Image data parameters
delta_z = delta_d/2;
x_axis = -image_width/2 : delta_x : image_width/2;          % Define image x-axis (lateral)
z_axis = z_start: delta_z : z_stop;                         % Define image z-axis (axial)
Beamformed_DATA = zeros(length(z_axis), length(x_axis));    % Allocate memeory for the Beamformed Data
% Calculate the diagonal length as the maximum image dimension (in samples)
max_dimension = round(sqrt(image_width^2 + z_stop^2)/(delta_z)); 
% Zero padding according to the maximum image dimension
RF_data128_padded = [RF_data128;  zeros(max_dimension-receive_depth, N_elements)];
% Delay calculations and Beamforming
for channel = 1 : N_elements            % Beamform every channel
    channel_location = channel*pitch;
    
    for zj = 1 : length(z_axis)         % Calculate the beamformed data for each pixel (depth)
    TX_delay = zj*delta_z;              % Distance from the transmitter to the points
        for xj = 1 : length(x_axis)     % Calculate the beamformed data for each pixel (lateral)
            RX_delay =sqrt(zj^2 + (xj-x)^2)*delta_x  ;  % Distance from the point to the receiver
            RF_address = round((TX_delay + RX_delay)/delta_d);
            Beamformed_DATA(zj, xj) = Beamformed_DATA(zj, xj) + RF_data128_padded(RF_address, channel); 
        end    
    end
end
toc