SPCE 5025 - Homework 3

Using Homework 2 as a base, and extending it to meet homework 3 requirements

Homework 2: https://github.com/gurbaflurb/spce5025_homework2

Takes in a provided set of vectors, one for the position, and one for the velocity of a satellite. This then calculates the following:

# First Vector
- Computes and outputs all 6 keplarian elements
- Computes the ECI -> UVW transformation
- Shows that the transform multiplied by vector r1 - transform multiplied by vector r2 = the transform multiplied by (r1 - r2)


# Second Vector
- Computes and outputs all 6 keplarian elements

# Inverse Question
Given the following keplarian elements, compute the corresponding ECI position and velocity vectors
- semi-major axis: 7800000.0 meters
- eccentricity: 0.001
- inclination: 98.6 degrees
- RAAN: 30.0 degrees
- argument of periapsis: 40 degrees
- nu: 30.087853 degrees



As usual, I've pre staged all the provided vectors into a single yaml file to streamline things.
