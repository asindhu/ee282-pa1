## Provide your own options or compiler here.
#CFLAGS += -O3 -ipo -axP -xW -funroll-loops -g
CFLAGS += -g
CC = gcc
#CC = icc

## You shouldn't need to edit anything past this point.

APP = matmul
SRCS = driver.c matmul.c utils.c

## Check for the PAPI header file. We assume that if we have the
## header, we have the library as well.
ifneq ($(shell ls /filer1/vol3/class/ee282-spr11/papi/include/papi.h 2> /dev/null),)
  LDFLAGS = -lpapi -L/filer1/vol3/class/ee282-spr11/papi/lib
  CFLAGS += -DPAPI -I/filer1/vol3/class/ee282-spr11/papi/include
endif

all: $(APP)

$(APP): $(SRCS:.c=.o)
	$(LINK.c) $^ $(LOADLIBES) $(LDLIBS) -o $@

clean:
	rm -f *.o *~ $(APP)

# Some generate dependencies.
%.o: %.c utils.h Makefile
	$(COMPILE.c) $(OUTPUT_OPTION) $<
