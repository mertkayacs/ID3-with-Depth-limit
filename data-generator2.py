import sys
import random
from datetime import datetime
import statistics
class Generator:
    def __init_bounds(self, feature_values_size):
        self.features = ["F"+str(i+1) for i in range(len(feature_values_size))]
        self.feature_values = {}
        self.max_samples = 1
        self.weight = {}
        self.power = {}
        i=0
        for f in self.features:
            self.feature_values[f] = [(j+1) for j in range(feature_values_size[i])]
            self.max_samples *= len(self.feature_values[f])
            self.weight[f] = random.randint(0,4)
            self.power[f] = random.randint(1,4)
            i+=1
        print("Weight:")
        print(self.weight)
        print("Power:")
        print(self.power)
        
    def evaluate(self, row):
        return sum([self.weight.get(f) * pow(row.get(f), self.power.get(f)) for f in self.features])
        

    def __init__(self,num_of_samples, feature_values_size):
        self.id = datetime.today().strftime('%Y-%m-%d-%H.%M.%S')
        self.__init_bounds(feature_values_size)
        self.num_of_samples = num_of_samples
        if self.max_samples < num_of_samples:
            print("MAX SAMPLES SHOULD BE {:d}".format(self.max_samples))
            exit(0)
        
    def save(self):
        matrix = []
        exists_map = {}
        remaining=self.num_of_samples
        result_list = []
        while remaining > 0:
            selected = [str(self.num_of_samples-remaining+1)]
            will_add = ""
            row = {}
            for f in self.features:
                item = self.feature_values[f][random.randint(0,len(self.feature_values[f])-1)]
                row[f] = item
                selected.append(str(item))
                will_add+=str(item)
            if exists_map.get(will_add) != None:
                continue
            result = self.evaluate(row)
            result_list.append(result)
            selected.append(result)
            exists_map[will_add] = True
            matrix.append(selected)
            remaining-=1
        w = open("data-"+self.id+".csv",'w')
        w.write(","+",".join(self.features)+",target\n")
        mean = statistics.mean(result_list)
        
        for m in matrix:
            last = m[len(m)-1]
            if last < mean:
                m[len(m)-1] = "0"
            else:
                m[len(m)-1] = "1"
            w.write(",".join(m)+"\n")
        w.close()

if len(sys.argv) < 3:
    print("Usage: python dataset_generator2.py <num_of_samples> ...feature_values_size")
else:
    Generator(int(sys.argv[1]),[int(i) for i in sys.argv[2:]]).save()