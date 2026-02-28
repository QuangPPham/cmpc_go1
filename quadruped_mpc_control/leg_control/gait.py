import numpy as np
DTYPE = np.float32

class Gait(object):
    def __init__(self,
                 MPC_horizon: int,                    # MPC_horizon
                 MPC_segments:int,                    # MPC segments for each cycle (e.g 10)
                 offsets:np.ndarray | list,           # offsets in MPC segments (for each leg), e.g [0, 5, 5, 0] for trotting
                 stance_durations:np.ndarray | list,  # duration of stance phase (for each leg), e.g [5, 5, 5, 5] for trotting
                 gait_name:str | None):
        
        """
        offsets [0, 5, 5, 0] means that FR (1) and RL (2) legs will be 5 segments later than the other 2
        We assume 0 offset means it'll be a stance leg at the 0th segment for stance_duration segments
        and an offset of 5 means it'll be a stance leg at the 5th segment for stance_duration segments
        """

        self._mpc_table = np.zeros(MPC_horizon * 4, dtype=DTYPE)
        self._mpc_horizon = MPC_horizon
        self._nIterations = MPC_segments

        self._offsets = np.asarray(offsets, dtype=DTYPE)
        self._durations = np.asarray(stance_durations, dtype=DTYPE)
        self._gaitName = gait_name

        # duration of stance and swing phase
        self._stance_duration = stance_durations[0]
        self._swing_duration = MPC_segments - self._stance_duration
        # offset in phase (0 to 1) for each leg
        self._PhaseOffsets = self._offsets / MPC_segments
        # stance duration in phase (0 to 1) for each leg
        self._Stance_Duration_phase = self._durations / MPC_segments

        self._iteration = 0 # step inside one cycle
        self._phase = 0 # current gait phase

    def setIterations(self, stepsBetweenMPC, currentStep):
        """Set what MPC segment and leg phase we are on based on current control iteration
        Assume that the period of each phase is exactly the horizon length
        """
        # how many MPC steps have we taken (bound from 0 to self._nIterations)
        self._iteration = np.floor(currentStep / stepsBetweenMPC) % self._nIterations
        # a phase of 1 means we have taken stepsBetweenMPC * self._nIterations steps
        self._phase = currentStep % (stepsBetweenMPC * self._nIterations) / (stepsBetweenMPC * self._nIterations)

    def getStanceTime(self, dt_MPC):
        """Get duration for leg stance in seconds
        dt_MPC: duration between MPC updates
        """
        return dt_MPC * self._stance_duration
    
    def getSwingTime(self, dt_MPC):
        """Get duration for leg swing in seconds
        dt_MPC: duration between MPC updates
        """
        return dt_MPC * self._swing_duration

    def getLegPhases(self):
        """Get what phase each leg is on
        """
        temp =  self._PhaseOffsets + np.array([self._phase]*4)  # offset phase
        # ensure phase between 0 and 1
        leg_phase = np.where(temp > 1.0, temp-1.0, temp)
        return leg_phase

    def getStanceProgress(self):
        """Check stance progress of each leg
        """
        # see how long has it been since the leg has been in stance phase
        offset_phase = np.array([self._phase] * 4) - self._PhaseOffsets
        progress = np.zeros(4, dtype=DTYPE)

        for i in range(4):
            # make sure phase is between 0 and 1
            if offset_phase[i] < 0:
                offset_phase[i] += 1.0

            # if exceed stance duration, it's in swing, and so progress is 0
            # if not, calculate stance progress (percent of stance duration)
            if offset_phase[i] <= self._Stance_Duration_phase[i]:
                progress[i] = offset_phase[i] / self._Stance_Duration_phase[i]

        return progress

    def getSwingProgress(self):
        """Check swing progress of each leg
        """
        # swing start = (stance) phase_offsets + stance_duration_phase
        swing_offset = self._PhaseOffsets + self._Stance_Duration_phase
        swing_offset = np.where(swing_offset > 1.0, swing_offset-1.0, swing_offset) # ensure phase in 0 to 1 range
        
        # swing_duration in phase for each leg
        swing_duration_phase = 1.- self._Stance_Duration_phase

        # see how long has it been since the leg has been in swing phase
        offset_phase = np.array([self._phase]*4) - swing_offset
        progress = np.zeros(4, dtype=DTYPE)

        for i in range(4):
            # make sure phase is between 0 and 1
            if offset_phase[i] < 0: 
                offset_phase[i] += 1.0

            # if exceed swing duration, it's in stance, and so progress is 0
            # if not, calculate swing progress (percent of swing duration)
            if offset_phase[i] <= swing_duration_phase[i]:
                if swing_duration_phase[i] == 0:
                    progress[i] = 0.
                else:
                    progress[i] = offset_phase[i] / swing_duration_phase[i]

        return progress

    def getLegStates(self, iteration=None):
        """Return array of boolean with 1 being foot is in contact, and 0 being foot is in swing
        """
        if iteration is None:
            iteration = self._iteration
        progress = np.array([iteration]*4) - self._offsets
        state = np.zeros(4)
        for j in range(4):
            if progress[j] < 0:
                progress[j] += self._nIterations
            if progress[j] < self._durations[j]:
                state[j] = 1
            else:
                state[j] = 0
        return state

    def getMPCtable(self):
        """Construct MPC table
        """
        # _mpc_table is a 1d list with horizon*4 elements
        # where the (i*4 + j) element tells us the contact
        # state of the j-th leg in the i-th MPC segment
        for i in range(self._mpc_horizon):
            iter = (i + self._iteration) % self._nIterations
            state = np.asarray(self.getLegStates(iter), dtype=DTYPE)
            self._mpc_table[4*i : 4*(i+1)] = state
            
        return self._mpc_table

trotting = Gait(10, 10, 
                np.array([0, 5, 5, 0], dtype=DTYPE), 
                np.array([5, 5, 5, 5], dtype=DTYPE), "Trotting")    
        
bounding = Gait(10, 10,
                np.array([5, 5, 0, 0], dtype=DTYPE), 
                np.array([4, 4, 4, 4], dtype=DTYPE), "Bounding")
        
pronking = Gait(10, 10,
                np.array([0, 0, 0, 0], dtype=DTYPE), 
                np.array([4, 4, 4, 4], dtype=DTYPE), "Pronking")

pacing =   Gait(10, 10,
                np.array([5, 0, 5, 0], dtype=DTYPE), 
                np.array([5, 5, 5, 5], dtype=DTYPE), "Pacing")

galloping = Gait(10, 10,
                np.array([0, 2, 7, 9], dtype=DTYPE), 
                np.array([4, 4, 4, 4], dtype=DTYPE), "Galloping")

walking =  Gait(10, 10,
                np.array([0, 3, 5, 8], dtype=DTYPE), 
                np.array([5, 5, 5, 5], dtype=DTYPE), "Walking")

trotRunning =  Gait(10, 10,
                    np.array([0, 5, 5, 0], dtype=DTYPE), 
                    np.array([4, 4, 4, 4], dtype=DTYPE), "Trot Running")
