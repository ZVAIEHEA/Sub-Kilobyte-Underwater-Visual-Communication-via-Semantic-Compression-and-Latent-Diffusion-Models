import sys
import os
import torch
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import json

chemin_fishial = "/Users/paulo/Desktop/Semantic imagery/ZVAIEH14/fish-identification/weights"
if chemin_fishial not in sys.path:
    sys.path.append(chemin_fishial)

SEMANTIC_LABELS = {
    (255, 255, 0): "Fish",               
    (0, 255, 0): "Plant/Sea-grass",      
    (0, 0, 255): "Human divers",         
    (255, 0, 0): "Robot",                
    (255, 255, 255): "Sea-floor/Rock",   
    (0, 255, 255): "Wrecks/Ruins",       
    (0, 0, 0): "Background",             
    (255, 0, 255): "Reefs and invertebrates" 
}

from inference import FishInferenceEngine, InferenceConfig

def split_mask_by_gray_then_color(mask_path, display=False): #Séparer les masques binaires des objets en fonction de la valeur de gris et de la couleur
    img_rgb = Image.open(mask_path).convert('RGB')
    img_gray = img_rgb.convert('L')
    mask_rgb = np.array(img_rgb)
    mask_gray = np.array(img_gray)
    
    unique_grays = np.unique(mask_gray)
    unique_grays = unique_grays[unique_grays != 0]
    
    separated_masks = {}
    for gray_val in unique_grays:
        color_mask = np.zeros_like(mask_rgb)
        match_indices = (mask_gray == gray_val)
        color_mask[match_indices] = mask_rgb[match_indices]
        rgb_color = mask_rgb[match_indices][0]
        
        separated_masks[gray_val] = {
            'binary_mask': match_indices.astype(np.uint8) * 255, 
            'color_mask': color_mask,
            'rgb_color': list(rgb_color)
        }
    return separated_masks

def generate_contour_points(separated_masks, kernel_size=5, min_area=5000): #Générer les contours des objets à partir des masques binaires
    contours_dict = {}
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    for gray_val, data in separated_masks.items():
        binary_mask = data['binary_mask']
        smoothed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)
        
        raw_contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        valid_contours = [c for c in raw_contours if cv2.contourArea(c) > min_area]
        
        contours_dict[gray_val] = {
            'contours': valid_contours,
            'rgb_color': data['rgb_color']
        }
    return contours_dict

def store_contour_points(contours_dict, min_points=20): #Stocker les contours des objets dans un dictionnaire structuré
    stored_objects = {}
    for index, (gray_val, data) in enumerate(contours_dict.items()):
        object_name = f"Object_{index + 1}"
        color = tuple(int(c) for c in data['rgb_color'])
        object_contours = []
        
        for contour in data['contours']:
            points = contour.reshape(-1, 2).tolist()
            if len(points) >= min_points:
                object_contours.append(points)
            
        if object_contours:
            stored_objects[object_name] = {
                'gray_value': int(gray_val),
                'rgb_color': color,
                'contour_points': object_contours
            }
    return stored_objects

def format_contours_by_components(saved_objects): #Formater les contours des objets en un dictionnaire structuré
    formatted_objects = {}
    for obj_name, obj_data in saved_objects.items():
        formatted_data = {
            'gray_scale': obj_data['gray_value'],
            'rgb_color': obj_data['rgb_color'],
            'contours': {}
        }
        for index, points_list in enumerate(obj_data['contour_points']):
            contour_name = f"contour_{index + 1}"
            formatted_data['contours'][contour_name] = points_list
            
        formatted_objects[obj_name] = formatted_data
    return formatted_objects

def get_average_color_in_contour(original_img_array, contour_points): #Calculer la couleur moyenne à l'intérieur d'un contour donné
    mask = np.zeros(original_img_array.shape[:2], dtype=np.uint8)
    pts = np.array(contour_points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    mean_val = cv2.mean(original_img_array, mask=mask)
    return (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))

def get_ocean_color(original_img_array, formatted_objects): #Calculer la couleur moyenne de l'océan en utilisant les contours des objets
    h, w = original_img_array.shape[:2]
    mask_objets = np.zeros((h, w), dtype=np.uint8)
    

    for obj_data in formatted_objects.values():
        for points in obj_data['contours'].values():
            if points:
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask_objets, [pts], 255)

    mask_fond = cv2.bitwise_not(mask_objets)
    bg_color_rgb = cv2.mean(original_img_array, mask=mask_fond)[:3]
    return (int(bg_color_rgb[0]), int(bg_color_rgb[1]), int(bg_color_rgb[2]))

def process_objects_semantics(original_image_path, formatted_objects, engine_path): #Enrichir les objets avec des informations sémantiques et identifier les espèces de poissons
    enriched_objects = formatted_objects.copy()
    original_img_pil = Image.open(original_image_path).convert("RGB")
    original_img_array = np.array(original_img_pil)
    
    poissons_presents = any(
        tuple(data['rgb_color'][:3]) == (255, 255, 0) for data in enriched_objects.values()
    )
    
    engine = None
    if poissons_presents:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            engine = FishInferenceEngine.from_bundle(
                bundle_path=engine_path, input_size=(154, 434), device=device,
                config=InferenceConfig(max_unique_classes=1)
            )
            engine.warmup(1)
        except Exception as e:
            print(f"[ERREUR IA] Impossible de charger Fishial : {e}")

    for obj_name, obj_data in enriched_objects.items():
        couleur_masque = tuple(obj_data['rgb_color'][:3])
        classe_globale = SEMANTIC_LABELS.get(couleur_masque, "Unknown")
        
        largest_contour_name = None
        max_points = 0
        for contour_name, points in obj_data['contours'].items():
            if len(points) > max_points:
                max_points = len(points)
                largest_contour_name = contour_name
                
        if largest_contour_name and max_points > 0:
            poly_points = obj_data['contours'][largest_contour_name]
            couleur_moyenne = get_average_color_in_contour(original_img_array, poly_points)
            enriched_objects[obj_name]['average_real_color'] = couleur_moyenne
            
            if classe_globale == "Fish" and engine:
                result = engine.predict_single(image=original_img_pil, poly=poly_points, method="natural_centroid")
                if result and result.best:
                    enriched_objects[obj_name]['species'] = result.best.name
                    enriched_objects[obj_name]['confidence'] = round(float(result.best.accuracy), 4)
                else:
                    enriched_objects[obj_name]['species'] = "Inconnu"
                    enriched_objects[obj_name]['confidence'] = 0.0
            else:
                enriched_objects[obj_name]['species'] = classe_globale
                enriched_objects[obj_name]['confidence'] = 1.0 
                
    return enriched_objects

def order_contour_points(formatted_objects): #Ordonner les points des contours pour chaque objet en utilisant l'algorithme du plus proche voisin
    ordered_objects = {}
    for obj_name, obj_data in formatted_objects.items():
        ordered_data = {k: v for k, v in obj_data.items() if k != 'contours'}
        ordered_data['contours'] = {}
        for contour_name, points in obj_data['contours'].items():
            if not points:
                ordered_data['contours'][contour_name] = []
                continue
            unvisited = list(points)
            ordered_points = [unvisited.pop(0)]
            while unvisited:
                current_point = ordered_points[-1]
                min_dist_idx = min(range(len(unvisited)), key=lambda i: (current_point[0] - unvisited[i][0])**2 + (current_point[1] - unvisited[i][1])**2)
                ordered_points.append(unvisited.pop(min_dist_idx))
            ordered_data['contours'][contour_name] = ordered_points
        ordered_objects[obj_name] = ordered_data
    return ordered_objects

def compute_fourier_descriptors(ordered_objects): #Calculer les descripteurs de Fourier pour chaque contour d'objet
    fourier_objects = {}
    for obj_name, obj_data in ordered_objects.items():
        fourier_data = {k: v for k, v in obj_data.items() if k != 'contours'}
        fourier_data['contours'] = {}
        for contour_name, sequence in obj_data['contours'].items():
            if not sequence: continue
            arr = np.array(sequence)
            complex_points = arr[:, 0] + 1j * arr[:, 1]
            fourier_data['contours'][contour_name] = np.fft.fft(complex_points).tolist()
        fourier_objects[obj_name] = fourier_data
    return fourier_objects

def compress_fourier_descriptors(fourier_objects, num_keep): #Compresse les coefficients de Fourier 
    compressed_objects = {}
    for obj_name, obj_data in fourier_objects.items():
        compressed_data = {k: v for k, v in obj_data.items() if k != 'contours'}
        compressed_data['contours'] = {}
        for contour_name, coeffs in obj_data['contours'].items():
            N = len(coeffs)
            target_coeffs = coeffs if N <= 2 * num_keep else coeffs[:num_keep] + coeffs[-num_keep:]
            compressed_data['contours'][contour_name] = {
                'original_length': N,
                'coeffs': [complex(int(round(c.real)), int(round(c.imag))) for c in target_coeffs]
            }
        compressed_objects[obj_name] = compressed_data
    return compressed_objects
class ComplexEncoder(json.JSONEncoder): 
    def default(self, obj):
        if isinstance(obj, complex):
            return f"{obj.real}+{obj.imag}j"
        return super().default(obj)

def estimate_payload_size(data): #Estimer la taille du dictionnaire compressé en octets pour le payload réseau
    
    dict_as_string = repr(data).replace(" ", "")
    return len(dict_as_string.encode('utf-8'))

def evaluate_dataset(images_dir, masks_dir, engine_path, output_json_path): #Évaluer un ensemble d'images et de masques, puis générer un rapport de compression

    report = {}
    
    print(f"Scan dee : {images_dir}")
    
    for filename in os.listdir(images_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        base_name = os.path.splitext(filename)[0]
        img_path = os.path.join(images_dir, filename)
        

        mask_path = os.path.join(masks_dir, base_name + '.bmp')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(masks_dir, base_name + '.jpg')
            
        if not os.path.exists(mask_path):
            print(f"[IGNORÉ] Aucun masque trouvé pour {filename}")
            continue
            
        print(f"{filename}")

        original_size_bytes = os.path.getsize(img_path)
        

        try:
            masks = split_mask_by_gray_then_color(mask_path)
            raw_contours_data = generate_contour_points(masks, min_area=5000)
            saved_objects = store_contour_points(raw_contours_data, min_points=50)
            final_structured_objects = format_contours_by_components(saved_objects)
            
            final_structured_objects = process_objects_semantics(img_path, final_structured_objects, engine_path)
            
            ordered_structured_objects = order_contour_points(final_structured_objects)
            fourier_descriptors = compute_fourier_descriptors(ordered_structured_objects)
            compressed_descriptors = compress_fourier_descriptors(fourier_descriptors, num_keep=5)
            
            original_img_array = np.array(Image.open(img_path).convert("RGB"))
            ocean_color = get_ocean_color(original_img_array, final_structured_objects)
            
            compressed_descriptors['Background'] = {
                'species': 'Background',
                'average_real_color': ocean_color
            }
            
            # 3. Calcul de la taille du Dictionnaire Compressé (Payload réseau)
            compressed_size_bytes = estimate_payload_size(compressed_descriptors)
            
            # 4. Calcul du ratio et ajout au rapport
            ratio = round(original_size_bytes / compressed_size_bytes, 2) if compressed_size_bytes > 0 else 0
            
            report[filename] = {
                'original_size_bytes': original_size_bytes,
                'compressed_size_bytes': compressed_size_bytes,
                'compression_ratio': f"1:{ratio}"
            }
            
            print(f" IMAGE INITIALE: {original_size_bytes} Bytes")
            print(f" Payload  : {compressed_size_bytes} Bytes")
            print(f"RATIO  : 1:{ratio}")
            
        except Exception as e:
            print(f"erreur {filename}: {e}")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, cls=ComplexEncoder)
        
    print(f"\nFini {output_json_path}")



if __name__ == '__main__':
    original_file = "/Users/paulo/Desktop/Semantic imagery/ZVAIEH14/SUIM/TEST/images/w_r_14_.jpg"
    mask_file = "/Users/paulo/Desktop/Semantic imagery/ZVAIEH14/SUIM/TEST/masks/w_r_14_.bmp"
    chemin_modele = "/Users/paulo/Desktop/Semantic imagery/ZVAIEH14/fish-identification/weights/model.pt"
    
    masks = split_mask_by_gray_then_color(mask_file)
    raw_contours_data = generate_contour_points(masks)
    saved_objects = store_contour_points(raw_contours_data)
    final_structured_objects = format_contours_by_components(saved_objects)
    
    final_structured_objects = process_objects_semantics(
        original_image_path=original_file,
        formatted_objects=final_structured_objects,
        engine_path=chemin_modele
    )


    ordered_structured_objects = order_contour_points(final_structured_objects)
    fourier_descriptors = compute_fourier_descriptors(ordered_structured_objects)
    compressed_descriptors = compress_fourier_descriptors(fourier_descriptors, num_keep=5)


    original_img_array = np.array(Image.open(original_file).convert("RGB"))
    ocean_color = get_ocean_color(original_img_array, final_structured_objects)
    
    compressed_descriptors['Background'] = {
        'species': 'Background',
        'average_real_color': ocean_color
    }

    print("\nDone : dico prêt ")
    print(compressed_descriptors)
    # dossier_images = "/Users/paulo/Desktop/Semantic imagery/ZVAIEH14/SUIM/TEST/images"
    # dossier_masques = "/Users/paulo/Desktop/Semantic imagery/ZVAIEH14/SUIM/TEST/masks"
    # chemin_modele = "/Users/paulo/Desktop/Semantic imagery/ZVAIEH14/fish-identification/weights/model.pt"
    # fichier_rapport = "compression_report.json"

    # evaluate_dataset(dossier_images, dossier_masques, chemin_modele, fichier_rapport)