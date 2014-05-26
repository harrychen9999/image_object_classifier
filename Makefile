CC=g++
#CFLAGS+=-g
CFLAGS+=`pkg-config --cflags opencv`
LDFLAGS+=`pkg-config --libs opencv`


.PHONY: all clean

main: main.o
	$(CC) -o main main.o $(LDFLAGS)


main.o: src/main.cpp
	$(CC) -c $(CFLAGS) $<


all: $(PROG)

clean:
	rm -f main main.o
