# Reachability 

class state(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __str__(self):
        return "S("+str(self.x)+","+str(self.y)+")"
    def __repr__(self):
        return str(self)
    def __hash__(self):
        return hash(self.__repr__())
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y
        else:
            return False
    def __ne__(self, other):
        return (not self.__eq__(other))

    def x(self):
        return self.x
    def y(self):
        return self.y 

class obstacle(object):
    def __init__(self,name,states):
        self.name = name
        self.states = states
    def __str__(self):
        return str(self.name)+"("+str(self.states)+")"
    def __repr__(self):
        return str(self)
    def __hash__(self):
        return hash(self.__repr__())
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name and self.states == other.states
        else:
            return False
    def __ne__(self, other):
        return (not self.__eq__(other))

    def name(self):
        return self.name
    def states(self):
        return self.states


