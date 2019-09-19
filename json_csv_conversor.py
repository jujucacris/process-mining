import json
import json_handler
import sys

def write_csv(f, filename):
    filename = filename.replace(".json", ".csv")
    c = open(filename, "w+")
    c.write("traceid,label,activity,user\n")
    i = 1
    for case in f['cases']:
        for event in case['events']:
            if case['attributes']['label'] != "normal":
                label = "anomaly"
            else:
                label = "normal"
            c.write("%d,%s,%s,%s\n" % (i, label, event['name'], event['attributes']['user']))
        i = i + 1

if __name__ == "__main__":

    for arg in sys.argv:
        if arg == sys.argv[0]:
            continue
        else:
            f = open(arg, "r")
            f = json_handler.json_load_byteified(f)

            write_csv(f, arg)
