CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O3 -march=native -fopenmp
LDFLAGS = -lopenblas -lm -lpthread

SOURCES = rescal.c svd.c utiles.c predition.c main_als.c
HEADERS = utiles.h svd.h rescal.h predition.h
OBJECTS = $(SOURCES:.c=.o)
EXECUTABLE = rescal

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(OBJECTS) $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

run_valgrind: $(EXECUTABLE)
	valgrind --leak-check=full --track-origins=yes ./$(EXECUTABLE)

.PHONY: all clean run_valgrind
