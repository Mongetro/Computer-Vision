all: detection_peau_humaine

detection_peau_humaine: detection_peau_humaine.o
	g++ detection_peau_humaine -o detection_peau_humaine `pkg-config --libs opencv` 

detection_peau.o: detection_peau_humaine.cpp
	g++ -c detection_peau_humaine `pkg-config --cflags opencv` 
	@echo 'Finished building target: $@'

clean:
	rm -rf *o detection_peau_humaine
