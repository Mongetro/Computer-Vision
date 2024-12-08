////****************************************************************************************************////
////                          	     Vision par Ordinateur     		   							        ////
////                               TP1: Détection de la peau					       			        ////
////																									////
////                                 Auteur: GOINT Mongetro                            		            ////
////                                 Promotion 22(2017-2019)                                			////
////																									////
////                                    Compilation:                                                    ////
////									1- make      		                        				 	////
////						   			2- ./detection_peau_humaine echelle seuil nom_image		        ////
////																									////
////		     DESCRIPTION: Ce programme permet de détecter la peau humaine dans des images 			////
////                                   																	////
////****************************************************************************************************////



#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


// DEFINITION DES CHEMINS D'ACCES AU REPERTOIRES PEAU ET NON-PEAU
#define NB_IMAGE 30
#define PATH_TO_NON_PEAU_IMAGES "base/non-peau/"
#define PATH_TO_PEAU_IMAGES "base/peau/"


using namespace cv;
using namespace std;


float** histogramme(string type, int echelle, float &nb_pixels) {

	float facteur_de_reduction = (float) echelle / 256;

	//CHEMIN DU REPERTOIRE DE L'IMAGE A CHOISIR
	char* PATH;

	if (type.compare("peau") == 0) {
		PATH = PATH_TO_PEAU_IMAGES;
	} else if (type.compare("non_peau") == 0) {
		PATH = PATH_TO_NON_PEAU_IMAGES;
	} else {
		cout << "Attention (repertoire érroné)! Vous avez entré le mauvais type de peau";
	}

	//LA MATRICE QUI DOIT CONTENIR L'IMAGE
	float ** histogramme;
	histogramme = new float*[echelle];
	for (int i = 0; i < echelle; i++) {
		histogramme[i] = new float[echelle];
		for (int j = 0; j < echelle; j++) {
			histogramme[i][j] = 0;
		}
	}

	//CONSTRUCTION DE L'HISTOGRAMME
	for (int i = 1; i <= NB_IMAGE; i++) {

		//ON ATTRIBUT LE NOM DE L'IMAGE
		char nom_image[50] = "";
		strcat(nom_image, PATH);
		char num[3] = "";
		sprintf(num, "%d", i);
		strcat(nom_image, num);
		strcat(nom_image, ".jpg");

		//ON CHARGE L'IMAGE DANS LE PROGRAMME
		Mat image;
		image = imread(nom_image, 1);

		if (!image.data) {
			cout << "Image non valide " << endl;
			exit(0);
		} else {

			// ICI ON CONVERTIR L'IMAGE 
			Mat resultat;
			cvtColor(image, resultat, CV_BGR2Lab);

			// ON PARCOUR L'IMAGE POUR REMPLIR L'HISTOGRAMME
			for (int k = 0; k < resultat.rows; k++) {
				for (int l = 0; l < resultat.cols; l++) {

					// CHOIX DES VALEURS a ET b
					int a = resultat.at<Vec3b>(k, l).val[1]
							* facteur_de_reduction;
					int b = resultat.at<Vec3b>(k, l).val[2]
							* facteur_de_reduction;

					// ON MET A JOUR LES VALEUS DE L'HISTOGRAMME
					if (image.at<Vec3b>(k, l) != Vec3b(0, 0, 0)) {

						histogramme[a][b] = histogramme[a][b] + 1;
					}
				}
			}
		}
	}

	// AMELIORTION DE LA DETECTION PAR LE LISSAGE DE L'HISTOGRAMME

			for (int i = 1; i < (echelle - 1); i++) {
				for (int j = 1; j < (echelle - 1); j++) {
					histogramme[i][j] = histogramme[i][j]
							+ (histogramme[i - 1][j - 1] + histogramme[i - 1][j]
									+ histogramme[i - 1][j + 1] + histogramme[i][j - 1]
									+ histogramme[i][j + 1] + histogramme[i + 1][j - 1]
									+ histogramme[i + 1][j] + histogramme[i + 1][j + 1])
									/ 8;
				}
			}

	//ICI ON NORMALISE L'HISTOGRAMME
	for (int m = 0; m < echelle; m++) {
		for (int n = 0; n < echelle; n++) {
			if(histogramme[m][n] !=0)
				nb_pixels += histogramme[m][n];

		}
	}

	for (int m = 0; m < echelle; m++) {
			for (int n = 0; n < echelle; n++) {
				if(histogramme[m][n] !=0)
								histogramme[m][n] /= nb_pixels;

			}
		}



	return histogramme;
}

//ON EVALUE LA PERFORMANCE DU PROGRAMME

void evaluation(Mat image_reference, Mat image_detectee) {

	int nb_pixels_peau_vrai = 0;
	int nb_pixels_peau_faux_pos = 0;
	int nb_pixels_peau_image_reference = 0;
	int nb_pixels_peau_faux_neg = 0;
	float performance;

	for (int i = 0; i < image_detectee.rows; i++) {
		for (int j = 0; j < image_detectee.cols; j++) {

			Vec3b Resultat = image_detectee.at<Vec3b>(i, j);
			Vec3b original = image_reference.at<Vec3b>(i, j);
			// CALCUL DU NOMBRE DE PIXELS PEAU DETECTE CORRECTEMENT DANS LE RESULTAT
			if (Resultat != Vec3b(0, 0, 0) && original != Vec3b(0, 0, 0)) {

				nb_pixels_peau_vrai++;
			}
			// CALCUL DU NOMBRE DE PIXELS PEAU MAL DANS LE RESULTAT
			if (Resultat != Vec3b(0, 0, 0) && original == Vec3b(0, 0, 0)) {
				nb_pixels_peau_faux_pos++;
			}
			// CALCUL DU NOMBRE DE PIXELS PEAU DETECTE DANS L'IMAGE DE REFFERENCE
			if (original != Vec3b(0, 0, 0)) {
				nb_pixels_peau_image_reference++;
			}
		}
	}

	nb_pixels_peau_faux_neg = nb_pixels_peau_image_reference -nb_pixels_peau_vrai;
	if(nb_pixels_peau_faux_neg < 0.0)
		nb_pixels_peau_faux_neg=0.0;

//ON CALCULE EN POURCENTAGE LA PERFORMANCE DU PROGRAMME
	performance = (float)nb_pixels_peau_vrai/(nb_pixels_peau_vrai
					+nb_pixels_peau_faux_pos + nb_pixels_peau_faux_neg);
	cout << "reference :"<<nb_pixels_peau_image_reference<< endl;
	cout << "correct :"<<nb_pixels_peau_vrai<< endl;
	cout << "faux_positif :"<<nb_pixels_peau_faux_pos<< endl;
	cout << "faux_negatif :"<<nb_pixels_peau_faux_neg<< endl;

	cout << "Perfomance du programme = " << performance * 100 << " %" << endl;

}

// DETECTION DE LA PEAU PAR LA METHDE SIMPLE
Mat detection_peau_simple(float** histo_peau, float** histo_non_peau,
		Mat image_test, int echelle) {

	float facteur_de_reduction = (float) echelle / 256;

	// ICI ON CONVERTI L'IMAGE DANS L'ESPACE LAB
	Mat resultat;
	cvtColor(image_test, resultat, CV_BGR2Lab);

	Mat masque(image_test.rows, image_test.cols, CV_8UC1);
	masque = Scalar(0);
	Mat sortie;
	image_test.copyTo(sortie);
	for (int k = 0; k < resultat.rows; k++) {
		for (int l = 0; l < resultat.cols; l++) {

			// CHOIX DS VALEURS a ET  b
			int a = resultat.at<Vec3b>(k, l).val[1] * facteur_de_reduction;
			int b = resultat.at<Vec3b>(k, l).val[2] * facteur_de_reduction;

			if (histo_peau[a][b] < histo_non_peau[a][b]) {

				sortie.at<Vec3b>(k, l) = Vec3b(0, 0, 0);

			} else {
				masque.at<uchar>(k, l) = 255;
			}
		}
	}

	imshow("image entree", image_test);

	imshow("masque", masque);
	imshow("sortie", sortie);

	return sortie;
}

// UTILISATION DES CALCULS DE PROBALITE POUR DETECTER LA PEAU
Mat detection_peau_bayes(float** histo_peau, float** histo_non_peau,
		Mat image_test, int echelle, float seuil, float nb_pixels_peau,
		float nb_pixels_non_peau) {

	float facteur_de_reduction = (float) echelle / 256;
	float proba_peau = 0.0;
	float proba_non_peau = 0.0;

	//PROBABILITE POUR PEAU ET NON-PEAU

	proba_peau = nb_pixels_peau / (nb_pixels_peau + nb_pixels_non_peau);
	proba_non_peau = nb_pixels_non_peau / (nb_pixels_peau + nb_pixels_non_peau);


	//CONVERSION DE L'IMAGE test DANS L'ESPACE LAB
	Mat resultat;
	cvtColor(image_test, resultat, CV_BGR2Lab);

	// CREATION DU MASQUE DE LA PEAU
	Mat masque(image_test.rows, image_test.cols, CV_8UC1);
	masque = Scalar(0);

	//CREATION DE L'IMAGE resultat
	Mat sortie;
	image_test.copyTo(sortie);

	for (int k = 0; k < resultat.rows; k++) {
		for (int l = 0; l < resultat.cols; l++) {

			// CHOIX DES VALEURS a ET b
			int a =0, b=0;
			 a = resultat.at<Vec3b>(k, l).val[1] * facteur_de_reduction;
			 b = resultat.at<Vec3b>(k, l).val[2] * facteur_de_reduction;
			 

			// PROBABILITE DE DECISION
			 float proba_decision = 0.0;
			 proba_decision = (histo_peau[a][b] * proba_peau)
					/ (histo_peau[a][b] * proba_peau
							+ histo_non_peau[a][b] * proba_non_peau);

			// LA MISE A JOUR DES VALEURS DE L'HISTOGRAMME
			if (proba_decision > seuil) {
				masque.at<uchar>(k, l) = 255;

			} else {
				sortie.at<Vec3b>(k, l) = Vec3b(0, 0, 0);
			}
		}

	}

	// ON POSSEDE AU POST-TRAITEMENT DE L'IMAGE
	int erosion_size = 1;
	int dilatation_size = 3;

	Mat dilate_element = getStructuringElement(MORPH_CROSS,
			Size(2 * dilatation_size + 1, 2 * dilatation_size + 1),
			Point(dilatation_size, dilatation_size));

	Mat erode_element = getStructuringElement(MORPH_CROSS,
			Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			Point(erosion_size, erosion_size));
	dilate(sortie, sortie, dilate_element);

	erode(sortie, sortie, erode_element);

	imshow("image entree", image_test);

	imshow("masque", masque);
	imshow("sortie", sortie);


	return sortie;

}

// AFFICHAGE DE L'ISTOGRAMME
void histogramme_print(float ** histogramme, int echelle, string type) {

	Mat big_histogramme(256, 256, CV_8UC1);
	float valeur_maximale = 0.0;

	// ON DETERMINE LA VALEUR MAXIMALE DE L'HISTOGRAMME

	for (int i = 0; i < echelle; i++) {
		for (int j = 0; j < echelle; j++) {
			if (histogramme[i][j] > valeur_maximale)
				valeur_maximale = histogramme[i][j];
		}
	}

	//Agrandissement, normalisation de la matrice de l'histogramme et transformation en image
	// ON AGRANDIT, ON NOORMALISE LA MATTRICE DE L'HISTOGRAMME ET ON TRANSFORME EN IMAGE

	for (int i = 0; i < echelle; i++) {
		for (int j = 0; j < echelle; j++) {
			for (int k = 0; k < 256/echelle; k++) {
				for (int l = 0; l < 256/echelle; l++)
					big_histogramme.at<uchar>(i * 256/echelle + k, j * 256/echelle + l) =
							saturate_cast<uchar>(
									((histogramme[i][j]) / valeur_maximale)
											* 255);
						}
		}
	}

	// ON ENREGISTRE L'ISTOGRAMME DANS LE REPERTOIRE DE DESTINATION
	char nom_histogramme[50] = "";
	strcat(nom_histogramme, "histogramme/histogramme_");
	if (type.compare("peau") == 0) {
		strcat(nom_histogramme, "peau");
	} else {
		strcat(nom_histogramme, "non-peau");
	}
	strcat(nom_histogramme, ".jpg");
	if (!imwrite(nom_histogramme, big_histogramme))
		cout << "Erreur lors de l'enregistrement" << endl;

	// ON AFFICHE L'HISTOGRAMME
	imshow(nom_histogramme, big_histogramme);
}

// ON DEFINIE LA FONCTION PRINCIPALE (main)
int main(int argc, char** argv) {

	int echelle = 0;
	float seuil = 0.0;
	echelle = atoi(argv[1]);
	seuil = atof(argv[2]);
	float ** histo_peau = NULL;
	float ** histo_non_peau = NULL;
	float nb_pixels_peau = 0;
	float nb_pixels_non_peau = 0;
	char* arg_nom = argv[3];
	char nom_image_test[50]= "";
	strcat(nom_image_test,"base/test/");
	strcat(nom_image_test,arg_nom);

	// ON POSSEDE A LA LECTURE DE L'IMAGE
	Mat image_entre;
	image_entre = imread(nom_image_test, 1);

	char nom_image_reference[50] = PATH_TO_PEAU_IMAGES;
	strcat(nom_image_reference,arg_nom);

	//ON POSSEE A LA LECTURE DE L'IMAGE DE REFERENCE
	Mat image_reference;
	image_reference = imread(nom_image_reference, 1);
	imshow("image reference", image_reference);


	Mat image_detectee;

	// ON POSSEDE AU CALCUL DES HISTOGRAMME
	histo_peau = histogramme("peau", echelle, nb_pixels_peau);

	histo_non_peau = histogramme("non_peau", echelle, nb_pixels_non_peau);


	image_detectee = detection_peau_bayes(histo_peau, histo_non_peau,
			image_entre, echelle, seuil, nb_pixels_peau, nb_pixels_non_peau);

	char nom_image_resultat[50] ="";
		strcat(nom_image_resultat,"resultat/");
		strcat(nom_image_resultat,"resultat_image_");
		strcat(nom_image_resultat,arg_nom);
		if (!imwrite(nom_image_resultat, image_detectee))
				cout << "Attention! Il y a erreur lors de l'enregistrement" << endl;

	evaluation(image_reference, image_detectee);
	histogramme_print(histo_peau,echelle,"peau");
	histogramme_print(histo_non_peau,echelle,"non_peau");
	waitKey(0);
	return 0;
}
