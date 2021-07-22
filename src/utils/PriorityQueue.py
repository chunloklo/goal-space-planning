class PriorityQueue(object):
    def __init__(self):
        self.queue = []
  
    def __str__(self):
        return ' '.join([str(i) for i in self.queue])
  
    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0
  
    # for inserting an element in the queue
    def insert(self, data):
        self.queue.append(data)

    def length(self):
        return len(self.queue)
  
    # for popping an element based on Priority
    def delete(self):
        try:
            max = 0
            for triple in self.queue:

                if int(triple[2]) > self.queue[max][2]:
                    
                    max = self.queue.index(triple)
            item = self.queue[max]
            del self.queue[max]
            return item
        except IndexError:
            print()
            exit()