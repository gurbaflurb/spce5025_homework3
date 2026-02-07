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

def convert_arbitrary_perifocal_to_eci(a, e, inclination, raan, aop, nu) -> tuple:
        '''Converts manually provided perifocal values to ECI coordinates. Uses radians and not degrees. Passing in degrees will mess up the calculation'''

        # Hard coded because a nice solution for this exists (Pulled from slide 74)
        x = [ math.cos(raan)*math.cos(aop) - math.sin(raan)*math.sin(aop)*math.cos(inclination), -math.cos(raan)*math.sin(aop) - math.sin(raan)*math.cos(aop)*math.cos(inclination), math.sin(raan)*math.sin(inclination)]
        y = [ math.sin(raan)*math.cos(aop)+math.cos(raan)*math.sin(aop)*math.cos(inclination), -math.sin(raan)*math.sin(aop)+math.cos(raan)*math.cos(aop)*math.cos(inclination), -math.cos(raan)*math.sin(inclination)]
        z = [ math.sin(inclination)*math.sin(aop), math.sin(inclination)*math.cos(aop), math.cos(inclination)]

        return (x, y, z)

def find_arbitrary_position_and_velocity_vector(a: float, eccentricity: float, nu: float) -> tuple:
    '''Returns a tuple in the form position_vector, velocity_vector'''
    mu = 398600441800000.0 # From WGS84
    
    perifocal = a*(1.0-math.pow(eccentricity, 2))
    radius = perifocal/(1.0 + eccentricity*math.cos(nu))

    return ([radius*math.cos(nu), radius*math.sin(nu), 0.0],
            [-math.sqrt(mu/perifocal)*math.sin(nu), math.sqrt(mu/perifocal)*(eccentricity+math.cos(nu)), 0.0])




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
        
        self.semi_major_axis = self.determine_semi_major_axis()
        
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


    def determine_semi_major_axis(self):
        return -(self.mu/(2 * self.energy))

    def determine_eccentricity(self):
        return np.linalg.norm(self.b_vector / self.mu)

    def determine_eccentricity_anomaly(self):
        n_e = np.dot(self.r_vector, self.r_dot_vector)/math.sqrt(self.mu*self.semi_major_axis)
        d_e = 1 - (np.linalg.norm(self.r_vector)/self.semi_major_axis)
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
        return 2 * math.pi * math.sqrt((math.pow(self.semi_major_axis, 3))/self.mu)

    def determine_apogee_radii(self):
        return self.semi_major_axis * (1 + self.energy)

    def determine_perigee_radii(self):
        return self.semi_major_axis * (1 - self.energy)

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
        print(f'Semi-major Axis       : {self.semi_major_axis} meters')
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
        return math.sqrt(self.mu/math.pow(self.semi_major_axis, 3))

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

        time_to_angle = math.sqrt(math.pow(self.semi_major_axis, 3)/self.mu) * ((2 * math.pi * perigee_passes) + pt_1 - pt_2)

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
        return self.semi_major_axis*(1-math.pow(self.eccentricity, 2))

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
    
    def convert_perifocal_to_equinoctial(self):
        '''Converts the Keplarian Elements for a given orbit and returns a tuple with all the Equinoctial values'''
        # Direct Set: 0 <= i < 180
        if self.inclination >= 0 and self.inclination < 180:
            semi_major_axis = self.semi_major_axis
            h = self.eccentricity * math.sin(self.aop + self.nu)
            k = self.eccentricity * math.cos(self.aop + self.nu)
            p = math.tan(self.inclination/2) * math.sin(self.nu)
            q = math.tan(self.inclination/2) * math.cos(self.nu)

            # Lambda, but not a python lambda function
            l = self.mean_anomaly + self.aop + self.nu



        # Retrograde Set: 0 < i <= 180
        elif self.inclination > 0 and self.inclination <= 180:
            semi_major_axis = self.semi_major_axis
            h = self.eccentricity * math.sin(self.aop - self.nu)
            k = self.eccentricity * math.cos(self.aop - self.nu)
            p = (1/math.tan(self.inclination/2)) * math.sin(self.nu)
            q = (1/math.tan(self.inclination/2)) * math.cos(self.nu)

            # Lambda, but not a python lambda function
            l = self.mean_anomaly + self.aop - self.nu

        return (semi_major_axis, h, k, p, q, l)

    def convert_coordinates_to_uvw(self) -> tuple:
        '''Convert given ECI coordinates (r_vector) and its cooresponding velocities (r_dot_vector) to UVW coordinates.'''
        u = self.r_vector/np.linalg.norm(self.r_vector)
        w = (np.cross(self.r_vector, self.r_dot_vector))/np.linalg.norm(np.linalg.cross(self.r_vector, self.r_dot_vector))
        v = np.cross(w, u)

        return (u, w, v)

    def convert_coordinates_to_lvlh(self) -> tuple:
        '''Convert given ECI coordinates (r_vector) and its cooresponding velocities (r_dot_vector) to LVLH coordinates.'''
        z = (- self.r_vector)/np.linalg.norm(self.r_vector)
        y = np.cross(self.r_dot_vector, self.r_vector)/np.linalg.norm(np.cross(self.r_dot_vector, self.r_vector))
        x = np.cross(y, z)

        return (x, y, z)

    def convert_perifocal_to_eci(self) -> tuple:
        '''Converts manually provided perifocal values to ECI coordinates. Uses radians and not degrees. Passing in degrees will mess up the calculation'''

        # Hard coded because a nice solution for this exists (Pulled from slide 74)
        x = [ math.cos(self.raan)*math.cos(self.aop) - math.sin(self.raan)*math.sin(self.aop)*math.cos(self.inclination), -math.cos(self.raan)*math.sin(self.aop) - math.sin(self.raan)*math.cos(self.aop)*math.cos(self.inclination), math.sin(self.raan)*math.sin(self.inclination)]
        y = [ math.sin(self.raan)*math.cos(self.aop) + math.cos(self.raan)*math.sin(self.aop)*math.cos(self.inclination), -math.sin(self.raan)*math.sin(self.aop)+math.cos(self.raan)*math.cos(self.aop)*math.cos(self.inclination), -math.cos(self.raan)*math.sin(self.inclination)]
        z = [ math.sin(self.inclination)*math.sin(self.aop), math.sin(self.inclination)*math.cos(self.aop), math.cos(self.inclination)]

        return (x, y, z)

    def rotate_uvw_about_x(self, uvw, angle):
        '''Pass in the angle in degrees you would like to rotate your plane '''
        radian_angle = math.radians(angle)

        rotation_matrix = np.matrix([[1.0, 0.0, 0.0],
                                     [0.0, math.cos(radian_angle), math.sin(radian_angle)],
                                     [0.0, -math.sin(radian_angle), math.cos(radian_angle)]])

        return np.dot(rotation_matrix, uvw)
    
    def rotate_uvw_about_y(self, uvw, angle):
        '''Pass in the angle in degrees you would like to rotate your plane '''
        radian_angle = math.radians(angle)

        rotation_matrix = np.matrix([[math.cos(radian_angle), 0, -math.sin(radian_angle)],
                                     [0, 1, 0],
                                     [math.sin(radian_angle), 0, math.cos(radian_angle)]])

        return np.dot(rotation_matrix, uvw)
    
    def rotate_uvw_about_z(self, uvw, angle):
        '''Pass in the angle in degrees you would like to rotate your plane '''
        radian_angle = math.radians(angle)

        rotation_matrix = np.matrix([[math.cos(radian_angle), math.sin(radian_angle), 0],
                                     [-math.sin(radian_angle), math.cos(radian_angle), 0],
                                     [0, 0, 1]])

        return np.dot(rotation_matrix, uvw)



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
    ke1_u, ke1_w, ke1_v = ke1.convert_coordinates_to_uvw()
    print(f'U     {ke1_u}')
    print(f'V     {ke1_v}')
    print(f'W     {ke1_w}')

    rotated_matrix = ke1.rotate_uvw_about_x(np.matrix([ke1_u,
                                                   ke1_v,
                                                   ke1_w]), 30)
    
    print(f'Rotated 30 degree UVW coordinates:')
    print(f'U     {rotated_matrix[0]}')
    print(f'V     {rotated_matrix[1]}')
    print(f'W     {rotated_matrix[2]}')

    print()

    print(f'----- Vector 2 -----')
    ke2 = KeplerianElements(vector_data['vectors'][f'vector2']['x_pos'],
                               vector_data['vectors'][f'vector2']['y_pos'],
                               vector_data['vectors'][f'vector2']['z_pos'],
                               vector_data['vectors'][f'vector2']['x_velocity'],
                               vector_data['vectors'][f'vector2']['y_velocity'],
                               vector_data['vectors'][f'vector2']['z_velocity'])
    ke2.print_ke()
    ke2_u, ke2_w, ke2_v = ke2.convert_coordinates_to_uvw()
    print(f'U     {ke2_u}')
    print(f'V     {ke2_v}')
    print(f'W     {ke2_w}')

    rotated_matrix = ke2.rotate_uvw_about_x(np.matrix([ke2_u,
                                                   ke2_v,
                                                   ke2_w]), 30)
    
    print(f'Rotated 30 degree UVW coordinates:')
    print(f'U     {rotated_matrix[0]}')
    print(f'V     {rotated_matrix[1]}')
    print(f'W     {rotated_matrix[2]}')

    print()

    ke_file = 'keplarian_elements.yaml'
    ke_data = read_in_yaml(ke_file)

    a = ke_data['keplarianElements']['ke1']['a']
    e = ke_data['keplarianElements']['ke1']['e']
    i = math.radians(ke_data['keplarianElements']['ke1']['i'])
    raan = math.radians(ke_data['keplarianElements']['ke1']['RAAN'])
    aop = math.radians(ke_data['keplarianElements']['ke1']['aop'])
    nu = math.radians(ke_data['keplarianElements']['ke1']['nu'])

    print('----- ECI position and Velocity from provided Keplarian elements -----')
    print(f'Semi-Major Axis        : {a}')
    print(f'Eccentricity           : {e}')
    print(f'Inclination            : {i} radians')
    print(f'RAAN                   : {raan} radians')
    print(f'Argument of Periapsis  : {aop} radians')
    print(f'Nu                     : {nu} radians')

    x, y, z = convert_arbitrary_perifocal_to_eci(a, e, i, raan, aop, nu)
    print('ECI Coordinates:')
    print(f'X: {x}')
    print(f'Y: {y}')
    print(f'Z: {z}')

    perifocal_vector, perifocal_velocity_vector = find_arbitrary_position_and_velocity_vector(a, e, nu)

    print(f'Perifocal Position {perifocal_vector}')
    print(f'Perifocal velocity {perifocal_velocity_vector}')

    # Convert into proper matrix format
    coordinates = np.matrix([x, y, z])
    
    position = coordinates * np.matrix([perifocal_vector]).transpose()
    velocity = coordinates * np.matrix([perifocal_velocity_vector]).transpose()
    print(f'Position Vector: {position.transpose()}')
    print(f'Velocity Vector: {velocity.transpose()}')




if __name__ == '__main__':
    main()
