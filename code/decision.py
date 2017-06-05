import numpy as np

STEER_ANGLE = 15
STUCK_COUNT = 20

class stack(list):
    def __init__(self, _max):
        self._max = _max

    def push(self, item):
        if len(self) == self._max:
            del self[0]
        self.append(item)


def is_stuck(Rover, prev_pos):
    if Rover.vel > 0.2:
        return False
    if len(prev_pos) < STUCK_COUNT:
        return False
    x_pos, y_pos = Rover.pos
    for p in prev_pos:
        x_prev, y_prev = p
        x_diff = int(x_pos) - int(x_prev)
        y_diff = int(y_pos) - int(y_prev)
        if x_diff != 0 or y_diff != 0:
            return False
    print("STUCK!!! STUCK!!!  STUCK!!!  STUCK!!!")
    return True

def get_n_smallest(nums, k):
    idx = np.argpartition(nums, k)
    return idx[:k]

def get_mode(nums):
    (values, counts) = np.unique(nums, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]

def get_hist(nums):
    return np.histogram(nums, bins=360, density=True)

def get_mode_hist(nums):
    (counts, values) = get_hist(nums)
    ind = np.argmax(counts)
    return values[ind]

def get_max_dist(angles, distances, avoid_angles): # likely
    not_found = True
    while not_found:
        (counts, values) = get_hist(distances)
        ind = np.argmax(counts)
        a_max = values[ind]
        max_dist_angle = angles[ind]
        print('max_dist_angle', max_dist_angle)
        print('avoid_angles', avoid_angles)
        if max_dist_angle in avoid_angles:
            del angles[ind]
            del distances[ind]
        else:
            not_found = False
    return max_dist_angle

def get_n_smallest_obst_distances(Rover, k):
    return get_n_smallest(Rover.dists['obstacles'], k)

def get_n_smallest_obst_angles(Rover, k):
    idxs = get_n_smallest_obst_distances(Rover, k)
    print("smallest obst distances:", Rover.dists['obstacles'][idxs])
    return Rover.angles['obstacles'][idxs] * 180/np.pi

def get_min_obst_dist_angle(Rover): # likely
    ind = np.argmin(Rover.dists['obstacles'])
    a_min = Rover.dists['obstacles'][ind]
    angles = Rover.angles['obstacles'] * 180/np.pi
    return (ind, a_min, angles[ind])

def get_min_rocks_dist_angle(Rover): # likely
    if len(Rover.dists['rocks']) == 0:
        return (None, None, None)
    ind = np.argmin(Rover.dists['rocks'])
    a_min = Rover.dists['rocks'][ind]
    angles = Rover.angles['rocks'] * 180/np.pi
    return (ind, a_min, angles[ind])

def from_forward_to_stop(Rover):
    # Set mode to "stop" and hit the brakes!
    Rover.throttle = 0
    # Set brake to stored brake value
    Rover.brake = Rover.brake_set
    Rover.steer = 0
    Rover.mode = 'stop'

def keep_braking(Rover, steer=0):
    Rover.throttle = 0
    Rover.brake = Rover.brake_set
    Rover.steer = steer

def get_steer_dir(Rover):
    #mode = get_mode(ints)
    #print('mode angles', mode)
    #med = np.median(ints)

    (rocks_ind, rock_dist, rocks_angle) = get_min_rocks_dist_angle(Rover)
    if is_rock_near(Rover):
        return rocks_angle

    close_obst_angles = get_n_smallest_obst_angles(Rover, 10)
    close_obst_ints = set(map(int, close_obst_angles))
    print("close_obst_angles:", close_obst_ints)

    angles = Rover.nav_angles * 180/np.pi
    nav_ints = list(map(int, angles))
    #print("nav_ints:", nav_ints)
    #clean_nav_ints = nav_ints - close_obst_ints

    #(obst_ind, a_min, obst_angle) = get_min_obst_dist_angle(Rover)
    #print('(obst_ind, a_min, obst_angle):', obst_ind, a_min, obst_angle)
    max_dist_angle = get_max_dist(nav_ints,
        Rover.nav_dists, close_obst_ints)

    #med = get_mode_hist(angles)
    #print('dists', ind, a_max, 'angle', angles[ind])
    #print('med angles', med)
    mean = np.mean(angles)
    print('mean angles', mean)
    #print('rocks', get_mode_hist(Rover.angles['rocks']))
    #print('obstacles', get_mode_hist(Rover.angles['obstacles']))
    return mean

def do_rock_stuff(Rover):
    (rocks_ind, rock_dist, rocks_angle) = get_min_rocks_dist_angle(Rover)

    Rover.mode = 'forward'
    if Rover.vel > 1.5:
        keep_braking(Rover, rocks_angle)
        print("rock BRAKING:", (rocks_ind, rock_dist, rocks_angle, Rover.vel))
    elif rock_dist > 10.0:
        forward_set(Rover)
    elif rock_dist < 10.0:
        print("rock NEAR SAMPLE:", (rocks_ind, rock_dist, rocks_angle, Rover.vel))
        keep_braking(Rover, rocks_angle)
        Rover.near_sample = True
  
def is_rock_near(Rover, dist=50.0):
    (rocks_ind, rock_dist, rocks_angle) = get_min_rocks_dist_angle(Rover)
    return rocks_angle is not None and rock_dist <= dist

def do_rover_steer(Rover):
    (rocks_ind, rock_dist, rocks_angle) = get_min_rocks_dist_angle(Rover)
    # Set steering to angle clipped to the range +/- 15
    if is_rock_near(Rover):
        do_rock_stuff(Rover)

    Rover.steer = np.clip(get_steer_dir(Rover),
        -STEER_ANGLE, STEER_ANGLE)

def forward_set(Rover):
    # If mode is forward, navigable terrain looks good 
    # and velocity is below max, then throttle 
    if Rover.vel < Rover.max_vel:
        # Set throttle value to throttle setting
        Rover.throttle = Rover.throttle_set
    else: # Else coast
        Rover.throttle = 0
    Rover.brake = 0

def do_rover_forward(Rover):
    forward_set(Rover)
    do_rover_steer(Rover)
  

prev_positions = stack(STUCK_COUNT)
# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    (rocks_ind, rock_dist, rocks_angle) = get_min_rocks_dist_angle(Rover)
    #stuck = is_stuck(Rover, prev_positions)
    #if stuck:
        #prev_positions[:] = []
        #from_forward_to_stop(Rover)

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 

            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
                do_rover_forward(Rover)
            elif len(Rover.nav_angles) < Rover.stop_forward:
                from_forward_to_stop(Rover)

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            if is_rock_near(Rover):
                do_rover_steer(Rover)
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                keep_braking(Rover)
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    print("HIIIIIIIIIIIII 1")
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    # TODO:
                    if Rover.steer > 0.0:
                        Rover.steer = STEER_ANGLE
                    else:
                        Rover.steer = -STEER_ANGLE # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    ## Set throttle back to stored value
                    #Rover.throttle = Rover.throttle_set
                    ## Release the brake
                    #Rover.brake = 0
                    ## Set steer to mean angle
                    #Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -STEER_ANGLE, STEER_ANGLE)
                    print("HIIIIIIIIIIIII 2")
                    do_rover_forward(Rover)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        print("DO ANYTHING?")
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    # add this position to the list of previous positions
    #prev_positions.push(Rover.pos)

    return Rover

