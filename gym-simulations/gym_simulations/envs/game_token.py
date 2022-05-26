class Token():
    """
    A Token object for use in passing_game.py.
    
    Attributes:
        state: "dropped" or "held" by an arm
        arm_id: Determines which bin it should be dropped in for a reward.
        location: current location of the token.
    """
    def __init__(self, entry_location, arm_id):
        self.state = 'dropped'
        self.location = entry_location
        self.arm_id = arm_id

    def set_state(new_state):
        self.state = new_state
        if self.state == 'dropped':
            # Dropped tokens are on the ground
            self.location[2] = 0.0
        assert (self.state == 'dropped' or self.state == 'held')

    def update_location(new_location):
        self.location = new_location
