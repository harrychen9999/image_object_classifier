CC=g++
CFLAGS+=-g
CFLAGS+=`pkg-config --cflags opencv`
LDFLAGS+=`pkg-config --libs opencv`

TARGETS = test_train sift train


.PHONY: all clean


sift: sift.o
	$(CC) -o sift sift.o $(LDFLAGS)

sift.o: src/sift.cpp
	$(CC) -c $(CFLAGS) $<

train: train.o
	$(CC) -o train train.o $(LDFLAGS)

train.o: src/train.cpp
	$(CC) -c $(CFLAGS) $<

test_train: test_train.o train.o
	$(CC) -o test_train test_train.o train.o $(LDFLAGS)

test_train.o: src/test_train.cpp
	$(CC) -c $(CFLAGS) $<

clean:
	rm -f *.o $(TARGETS)
