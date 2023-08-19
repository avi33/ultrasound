import torch
import torch.nn as nn

class DelayAndSum(nn.Module):
    def __init__(self, num_sensors, sampling_rate, speed_of_sound, angle_of_interest):
        super(DelayAndSum, self).__init__()
        
        self.num_sensors = num_sensors
        self.sampling_rate = sampling_rate
        self.speed_of_sound = speed_of_sound
        self.angle_of_interest = angle_of_interest
        
        self.sensor_positions = self.calculate_sensor_positions()
        self.delays = self.calculate_delays()
    
    def calculate_sensor_positions(self):
        # Assuming the ULA is linear and positioned along the x-axis
        # Sensor positions are equally spaced
        sensor_positions = torch.arange(0, self.num_sensors) * self.speed_of_sound / (2 * self.sampling_rate)
        return sensor_positions
    
    def calculate_delays(self):
        # Calculate delays based on the angle of interest
        delays = self.sensor_positions * torch.sin(self.angle_of_interest) / self.speed_of_sound
        return delays
    
    def forward(self, microphone_array):
        batch_size, num_samples, _ = microphone_array.size()
        
        # Initialize the output signal
        output_signal = torch.zeros(batch_size, num_samples)
        
        # Apply delay-and-sum beamforming
        for sensor_idx in range(self.num_sensors):
            delay_samples = torch.round(self.delays[sensor_idx] * self.sampling_rate).int()
            shifted_signal = torch.cat((microphone_array[:, delay_samples:], torch.zeros(batch_size, delay_samples)), dim=1)
            output_signal += shifted_signal
        
        return output_signal / self.num_sensors

    def update_angle(self, new_angle_of_interest):
        self.angle_of_interest = new_angle_of_interest
        self.delays = self.calculate_delays()


if __name__ == "__main__":
    # Example usage
    num_sensors = 128
    sampling_rate = 20e6  # Sample rate in Hz
    speed_of_sound = 343   # Speed of sound in m/s
    initial_angle = 0.5    # Initial angle of interest in radians

    beamformer = DelayAndSum(num_sensors, sampling_rate, speed_of_sound, initial_angle)

    # Update the angle of interest
    new_angle = 0.3  # New angle of interest in radians
    beamformer.update_angle(new_angle)

    # Now you can use the updated beamformer to process microphone array data
    # output_signal = beamformer(input_microphone_array)