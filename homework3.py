# Compute the Keplarian elements of the given vectors in vectors.yaml
# Compute the ECI->UVW transformation using the first vector
# Show that the transform multiplied by vector r1 - transform multiplied by vector r2 = the transform multiplied by (r1 - r2)
#

import math
import yaml

import numpy as np

# Read in a yaml that has all the initial vectors for position and velocity
def read_in_yaml(file_name):
    with open(file_name, 'r') as f:
        data = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return data


class KeplerianElements():
    '''
    Generates the Keplerian Elements given the 6 required parameters:
    X-Position, X-Velocity
    Y-Position, Y-Velocity
    Z-Position, Z-Velocity

    Depends on numpy for finding dot and cross products 
    '''
    def __init__(self, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel):
        self.initial_x_pos = x_pos
        self.initial_x_vel = x_vel
        self.initial_y_pos = y_pos
        self.initial_y_vel = y_vel
        self.initial_z_pos = z_pos
        self.initial_z_vel = z_vel

        self.mu = 398600441800000 # From WGS84

        z_hat = [0, 0, 1]

        self.r_vector = np.array([self.initial_x_pos, self.initial_y_pos, self.initial_z_pos])
        self.r_dot_vector = np.array([self.initial_x_vel, self.initial_y_vel, self.initial_z_vel])

        self.h_vector = self.determine_h()
        self.angular_momentum_vector = self.h_vector

        self.inclination = self.determine_inclination(z_hat)

        self.n_hat = self.determine_n_hat(z_hat)
        
        self.raan = self.determine_right_ascension_of_ascending_node()

        self.b_vector = self.determine_b()
        
        self.eccentricity = self.determine_eccentricity()

        self.energy = self.determing_energy()
        
        self.acceleration = self.determine_acceleration()
        
        self.orbital_period = self.determine_orbital_period()
        self.tp = self.orbital_period

        self.apogee_radii = self.determine_apogee_radii()

        self.perigee_radii = self.determine_perigee_radii()

        self.aop = self.determine_argument_of_periapsis()

        self.nu = self.determine_true_anomaly()
        self.true_anomaly = self.nu

        self.eccentricity_anomaly = self.determine_eccentricity_anomaly()

        self.mean_anomaly = self.determine_mean_anomaly()
        
        self.mean_motion = self.determine_mean_motion()

        self.v0 = self.determine_v_0()

        self.E0 = self.determine_E_0(self.eccentricity_anomaly)

        self.perifocal_positions = self.determine_perifocal_position()
        
        self.velocity_components = self.determine_velocity_components() 


    def determine_acceleration(self):
        return -(self.mu/(2 * self.energy))

    def determine_eccentricity(self):
        return np.linalg.norm(self.b_vector / self.mu)

    def determine_eccentricity_anomaly(self):
        n_e = np.dot(self.r_vector, self.r_dot_vector)/math.sqrt(self.mu*self.acceleration)
        d_e = 1 - (np.linalg.norm(self.r_vector)/self.acceleration)
        r = math.atan2(n_e, d_e)

        if np.dot(self.r_vector, self.b_vector) < 0:
            r = (2 * math.pi) + r

        return r
    
    def determine_arbitrary_eccentric_anomaly(self, angle):
        '''Takes in an angle in degrees'''
        radian_angle = math.radians(angle)

        return math.acos((self.eccentricity + math.cos(radian_angle))/(1 + self.eccentricity * math.cos(radian_angle)))

    def determine_inclination(self, z_hat: list):
        '''Returns in radians, convert to degrees if you need'''
        return math.acos(np.dot(self.h_vector, z_hat)/(np.linalg.norm(self.h_vector)))

    def determine_right_ascension_of_ascending_node(self):
        '''Returns in radians, convert to degrees if you need'''
        r = math.atan2(self.n_hat[1], self.n_hat[0])

        # Correct for if we are in quadrant 3 or 4 
        if r < 0:
            r = r + (2 * math.pi)

        return r

    def determine_true_anomaly(self):
        '''Returns in radians, convert to degrees if you need'''
        r = math.acos((np.dot(self.r_vector, self.b_vector))/(np.linalg.norm(self.r_vector) * np.linalg.norm(self.b_vector)))

        # Correct for if we are in quadrant 3 or 4
        if np.dot(self.r_vector, self.b_vector) < 0:
            r = (2 * math.pi) - r
        return r        

    def determine_argument_of_periapsis(self):
        '''Returns in radians, convert to degrees if you need'''
        return math.atan2(np.dot(self.h_vector/np.linalg.norm(self.h_vector), np.cross(self.n_hat, self.b_vector/np.linalg.norm(self.b_vector))), np.dot(self.n_hat, self.b_vector/np.linalg.norm(self.b_vector)))

    def determine_orbital_period(self):
        return 2 * math.pi * math.sqrt((math.pow(self.acceleration, 3))/self.mu)

    def determine_apogee_radii(self):
        return self.acceleration * (1 + self.energy)

    def determine_perigee_radii(self):
        return self.acceleration * (1 - self.energy)

    def determing_energy(self):
        return (math.pow(np.linalg.norm(self.r_dot_vector), 2)/2) - (self.mu/np.linalg.norm(self.r_vector))

    def determine_h(self):
        '''Returns the H-Hat, the cross product of the position vector (r) and the velocity vector (r-dot)'''
        return np.cross(self.r_vector, self.r_dot_vector)

    def determine_n_hat(self, z_hat: list):
        '''Returns the N-Hat'''
        return np.cross(z_hat, self.h_vector)/np.linalg.norm(np.cross(z_hat, self.h_vector))

    def determine_b(self):
        return np.cross(self.r_dot_vector, self.h_vector) - (self.mu * (self.r_vector/np.linalg.norm(self.r_vector))) 

    def print_ke(self):
        print(f'Position Vector       : {self.r_vector}')
        print(f'Velocity Vector       : {self.r_dot_vector}')
        print(f'Acceleration          : {self.acceleration} meters')
        print(f'Eccentricity          : {self.eccentricity}')
        print(f'Inclination           : {math.degrees(self.inclination)} Degrees')
        print(f'RAAN                  : {math.degrees(self.raan)} Degress')
        print(f'Argument of Periapsis : {math.degrees(self.aop)} Degrees')
        print(f'Nu                    : {math.degrees(self.nu)} Degrees')
        print(f'Nu                    : {self.nu} radians')
        print(f'Orbit Period          : {self.tp} seconds')
        print(f'Apogee Radii          : {self.apogee_radii} meters')
        print(f'Perigee Radii         : {self.perigee_radii} meters')

    def determine_mean_motion(self):
        return math.sqrt(self.mu/math.pow(self.acceleration, 3))

    def determine_mean_anomaly(self):
        return self.eccentricity_anomaly - self.eccentricity * math.sin(self.eccentricity_anomaly)

    def determine_v_0(self):
        return self.nu

    def determine_E_0(self, eccentric_anomaly):
        return math.atan2(math.sin(eccentric_anomaly), math.cos(eccentric_anomaly))
    
    def determine_time_of_flight(self, mean_anomaly):
        return mean_anomaly/self.mean_motion
    
    def determine_time_to_angle(self, angle, perigee_passes=0):
        '''Provided an angle from 0-360, returns the seconds to reach that angle from Nu, only works for 0-180'''
        E_1 = self.determine_arbitrary_eccentric_anomaly(angle)

        pt_1 = E_1 - self.eccentricity * math.sin(E_1)
        pt_2 = self.mean_anomaly

        time_to_angle = math.sqrt(math.pow(self.acceleration, 3)/self.mu) * ((2 * math.pi * perigee_passes) + pt_1 - pt_2)

        if time_to_angle < 0:
            time_to_angle = time_to_angle + 2 * math.pi
        
        return time_to_angle

    def determine_location_after_n_seconds(self, seconds, nu):
        '''Implemented with Keplers Equation'''
        cur_E0 = nu
        
        for i in range(0,10):
            cur_E0 = cur_E0 + (self.mean_motion * seconds + self.mean_anomaly - (cur_E0 - self.eccentricity * math.sin(cur_E0)))/(1 - self.eccentricity * math.cos(cur_E0))

        perigee_passes = math.trunc((cur_E0 - nu)/(2*math.pi))

        # Correct if we perform multiple orbits
        cur_E0 = cur_E0 % (2*math.pi)

        return (cur_E0, perigee_passes)
    
    def determine_true_anomaly_from_eccentric_anomaly(self, eccentric_anomaly):
        nu = math.atan2((math.sin(eccentric_anomaly)*math.sqrt(1-math.pow(self.eccentricity, 2)))/(1-self.eccentricity*math.cos(eccentric_anomaly)), (math.cos(eccentric_anomaly)-self.eccentricity)/(1-self.eccentricity*math.cos(eccentric_anomaly)))
        
        if nu < 0:
            nu = nu + 2*math.pi
        
        return nu

    def determine_p(self):
        return self.acceleration*(1-math.pow(self.eccentricity, 2))

    def determine_perifocal_position(self) -> list:
        x = (np.linalg.norm(self.r_vector) * math.cos(self.nu)).item()
        y = (np.linalg.norm(self.r_vector) * math.sin(self.nu)).item()

        return [x, y, 0]

    def determine_velocity_components(self) -> list:
        x_dot = -math.sqrt(self.mu/self.determine_p())*math.sin(self.nu)
        y_dot = (math.sqrt(self.mu/self.determine_p())*(self.eccentricity+math.cos(self.nu))).item()

        return [x_dot, y_dot, 0]

    def determine_f(self, nu, delta_nu):
        '''Takes in two angles, nu, and the delta nu as degrees'''
        radian_nu = math.radians(nu)
        radian_delta_nu = math.radians(delta_nu)

        r = self.determine_p()/(1+self.eccentricity*math.cos(radian_nu + radian_delta_nu))

        return 1 - (r/self.determine_p())*(1 - math.cos(radian_delta_nu))

    def determine_g(self, nu, delta_nu):
        '''Takes in two angles, nu, and the delta nu as degrees'''
        radian_nu = math.radians(nu)
        radian_delta_nu = math.radians(delta_nu)

        r = self.determine_p()/(1+self.eccentricity*math.cos(radian_nu + radian_delta_nu))

        return ((r*np.linalg.norm(self.r_vector))/(math.sqrt(self.mu * self.determine_p()))) * math.sin(radian_delta_nu)

    def determine_g_dot(self, delta_nu):
        '''Takes in the angle delta_nu as degrees'''
        radian_delta_nu = math.radians(delta_nu)

        return 1 - (np.linalg.norm(self.r_vector)/self.determine_p())*(1 - math.cos(radian_delta_nu))

    def determine_f_dot(self, nu, delta_nu):
        '''Takes in two angles, nu, and the delta nu as degrees'''
        radian_nu = math.radians(nu)
        radian_delta_nu = math.radians(delta_nu)

        r = self.determine_p()/(1+self.eccentricity*math.cos(radian_nu + radian_delta_nu))

        return math.sqrt(self.mu/self.determine_p()) * (math.tan(radian_delta_nu/2)) * (((1-math.cos(radian_delta_nu))/self.determine_p()) - (1/r) - (1/np.linalg.norm(self.r_vector)))

    def determine_new_velocity(self, f_dot, g_dot):
        '''Determine the new position vector at a new delta-v'''
        return np.array(self.perifocal_positions) * f_dot + np.array(self.velocity_components) * g_dot

    def determine_new_position(self, f, g):
        '''Determine the new position vector at a new delta-v'''
        return np.array(self.perifocal_positions) * f + np.array(self.velocity_components) * g



def main():

    vectors_file = 'vectors.yaml'
    vector_data = read_in_yaml(vectors_file)

    print(f'----- Vector 1 -----')
    ke1 = KeplerianElements(vector_data['vectors'][f'vector1']['x_pos'],
                               vector_data['vectors'][f'vector1']['y_pos'],
                               vector_data['vectors'][f'vector1']['z_pos'],
                               vector_data['vectors'][f'vector1']['x_velocity'],
                               vector_data['vectors'][f'vector1']['y_velocity'],
                               vector_data['vectors'][f'vector1']['z_velocity'])

    ke1.print_ke()

    print(f'E_0                           : {ke1.E0} Radians')
    print(f'E_0                           : {math.degrees(ke1.E0)} Degrees')

    print(f'v_0                           : {ke1.v0} Radians')
    print(f'v_0                           : {math.degrees(ke1.v0)} Degrees')

    print(f'Mean Motion (M)               : {ke1.mean_motion} radians')
    print(f'Mean Motion (M)               : {math.degrees(ke1.mean_motion)} Degrees')

    print(f'Mean Anomaly (n)              : {ke1.mean_anomaly}')
    print(f'Mean Anomaly (n)              : {math.degrees(ke1.mean_anomaly)} Degrees')

    print(f'Time of Flight from perigee   : {ke1.determine_time_of_flight(ke1.mean_anomaly)} seconds')

    print(f'Eccentic angle at 65 degrees  : {ke1.determine_arbitrary_eccentric_anomaly(65)}')
    print(f'Eccentic angle at 65 degrees  : {math.degrees(ke1.determine_arbitrary_eccentric_anomaly(65))} Degrees')

    print(f'Time to 65 Degrees            : {ke1.determine_time_to_angle(65)} seconds')

    time_to_angle_65, perigee_passes = ke1.determine_location_after_n_seconds(ke1.determine_time_to_angle(65), ke1.E0)
    print(f'Newton-Raphson (EA)           : {time_to_angle_65} radians')
    print(f'Newton-Raphson (EA)           : {math.degrees(time_to_angle_65)} Degrees')

    true_anomaly_65 = ke1.determine_true_anomaly_from_eccentric_anomaly(time_to_angle_65)
    print(f'Newton-Raphson (Nu)           : {true_anomaly_65} radians')
    print(f'Newton-Raphson (Nu)           : {math.degrees(true_anomaly_65)} Degrees')

    print()

    loc_after_2700, perigee_passes = ke1.determine_location_after_n_seconds(2700, ke1.E0)
    print(f'Location after 2700s (EA)       : {loc_after_2700} Radians')
    print(f'Location after 2700s (EA)       : {math.degrees(loc_after_2700)} Degrees')
    print(f'Perigee passes after 2700s (EA) : {perigee_passes}')

    print(f'Location after 2700s (Nu)       : {ke1.determine_true_anomaly_from_eccentric_anomaly(loc_after_2700)} Radians')
    print(f'Location after 2700s (Nu)       : {math.degrees(ke1.determine_true_anomaly_from_eccentric_anomaly(loc_after_2700))} Degrees')

    print()

    loc_after_2TP, perigee_passes = ke1.determine_location_after_n_seconds(2*ke1.tp, ke1.E0)
    print(f'Location after 2 TP, E0       : {loc_after_2TP} radians')
    print(f'Location after 2 TP, E0       : {math.degrees(loc_after_2TP)} Degrees')
    print(f'Perigee passes after 2TP, E0  : {perigee_passes}')

    print(f'Location after 2 TP, (Nu)     : {ke1.determine_true_anomaly_from_eccentric_anomaly(loc_after_2TP)} radians')
    print(f'Location after 2 TP, (Nu)     : {math.degrees(ke1.determine_true_anomaly_from_eccentric_anomaly(loc_after_2TP))} Degrees')
    print(f'Perigee passes after 2TP, (Nu): {perigee_passes}')

    print()

    loc_after_15000, perigee_passes = ke1.determine_location_after_n_seconds(15000, ke1.E0)
    print(f'Location after 15000s, (EA)        : {loc_after_15000} Radians')
    print(f'Location after 15000s, (EA)        : {math.degrees(loc_after_15000)} Degrees')
    print(f'Perigee passes after 15000s, (EA)  : {perigee_passes}')

    print(f'Location after 15000s, (Nu)        : {ke1.determine_true_anomaly_from_eccentric_anomaly(loc_after_15000)} Radians')
    print(f'Location after 15000s, (Nu)        : {math.degrees(ke1.determine_true_anomaly_from_eccentric_anomaly(loc_after_15000))} Degrees')
    print(f'Perigee passes after 15000s, (Nu)  : {perigee_passes}')
    print()



    print(f'----- Vector 2 -----')
    ke2 = KeplerianElements(vector_data['vectors'][f'vector2']['x_pos'],
                               vector_data['vectors'][f'vector2']['y_pos'],
                               vector_data['vectors'][f'vector2']['z_pos'],
                               vector_data['vectors'][f'vector2']['x_velocity'],
                               vector_data['vectors'][f'vector2']['y_velocity'],
                               vector_data['vectors'][f'vector2']['z_velocity'])
    
    print('Keplarian Elements')
    ke2.print_ke()

    print()

    perifocal_components = ke2.determine_perifocal_position()
    velocity_components = ke2.determine_velocity_components()
    print(f'Perifocal Components    : {perifocal_components}')
    print(f'Velocity Components     : {velocity_components}')

    print()

    print(f'p                       : {ke2.determine_p()}')
    print(f'f                       : {ke2.determine_f(math.degrees(ke2.nu), 33)} radians')
    print(f'g                       : {ke2.determine_g(math.degrees(ke2.nu), 33)} radians')
    print(f'f_dot                   : {ke2.determine_f_dot(math.degrees(ke2.nu), 33)} radians')
    print(f'g_dot                   : {ke2.determine_g_dot(33)} radians')

    print()

    print(f'r_arrow at delta-v     : {ke2.determine_new_position(ke2.determine_f(math.degrees(ke2.nu), 33), ke2.determine_g(math.degrees(ke2.nu), 33))}')
    print(f'v_arrow at delta-v     : {ke2.determine_new_velocity(ke2.determine_f_dot(math.degrees(ke2.nu), 33), ke2.determine_g_dot(33))}')
    

if __name__ == '__main__':
    main()
