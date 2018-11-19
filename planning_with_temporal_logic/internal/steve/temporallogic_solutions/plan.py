class InvalidIdSequence(Exception):
    def __init__(self, i, j):
        self.i = i
        self.j = j

    def __str__(self):
        return "id %d does not exist. Got id %d instead" % (i, j)

class Plan:
    def __init__(self, activities):
        if activities:
            self.activities = activities
        else:
            self.activities = []
        self.keys = []
        if len(self.activities) > 0:
            self.keys = self.activities[0].keys()
        
    def __eq__(self, other):

        def check_id(activities, i):
            if activities[i]["id"] != i:
                raise InvalidIdSequence(i, activities[i]["id"])

        if len(self.activities) != len(other.activities):
            return False
        if len(self.keys) != len(other.keys):
            return False
        for key in self.keys:
            if key not in other.keys:
                return False

        for i in range(len(self.activities)):
            check_id(self.activities, i)
            check_id(other.activities, i)
            for key in self.keys:
                if self.activities[i][key] != other.activities[i][key]:
                	#print self.activities[i][key], other.activities[i][key]
                	return False
        return True
    
    def extract(self, *actions, **kwargs): #, augment=None):
        if "augment" in kwargs:
            augment = kwargs["augment"]
        else:
            augment = None
        actions = set(actions)

        sequence = []
        for activity in self.activities:
            action = activity["activity"].split(None,1)[0].strip("()")
            if action in actions:
                if augment and augment in activity:
                    sequence.append((activity["activity"], activity[augment]))
                else:
                    sequence.append(activity["activity"])
        return sequence