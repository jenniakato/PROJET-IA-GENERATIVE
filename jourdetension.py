#!/usr/bin/env python
"""
Jourdetension.py
─────────────────────────────
Agent pour récupérer les jours de tension RTE (Tempo EDF)
Version améliorée avec meilleure gestion des erreurs
"""

import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import json


class JourDeTensionRTE:
    """Agent pour interroger l'API RTE sur les jours de tension électrique (Tempo)"""
    
    BASE_URL = "https://digital.iservices.rte-france.com/open_api/tempo_like_supply_contract/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize avec une clé API RTE (optionnelle pour démo)
        
        Args:
            api_key: Clé API RTE (obtenir sur https://data.rte-france.com/)
        """
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}" if api_key else "",
            "Content-Type": "application/json"
        }
        self.mode_demo = not bool(api_key)
        
        if self.mode_demo:
            print("⚠️  Mode démonstration activé (pas de clé API RTE)")
    
    def get_jour_tension(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère le statut de tension pour une date donnée
        
        Args:
            date: Date au format 'YYYY-MM-DD'. Si None, utilise aujourd'hui
            
        Returns:
            Dict contenant les informations de tension:
            - date: Date concernée
            - couleur_tempo: 'rouge', 'blanc', ou 'bleu'
            - niveau_tension: 'haute', 'moyenne', ou 'normale'
            - description: Description détaillée
            - mode: 'demo' ou 'api'
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Validation du format de date
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return {
                "error": "Format de date invalide",
                "message": "Utilisez le format YYYY-MM-DD",
                "date": date
            }
        
        # Mode démo si pas de clé API
        if self.mode_demo:
            return self._demo_response(date)
        
        # Appel API réel
        return self._api_call(date)
    
    def _api_call(self, date: str) -> Dict[str, Any]:
        """Effectue l'appel API réel vers RTE"""
        try:
            response = requests.get(
                f"{self.BASE_URL}/tempo_like_calendars",
                headers=self.headers,
                params={"start_date": date, "end_date": date},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Parsing de la réponse RTE
            if data and len(data) > 0:
                jour_data = data[0]
                return {
                    "date": date,
                    "couleur_tempo": jour_data.get("tempo_like_value", "bleu").lower(),
                    "niveau_tension": self._map_couleur_niveau(
                        jour_data.get("tempo_like_value", "bleu")
                    ),
                    "description": self._get_description(
                        self._map_couleur_niveau(jour_data.get("tempo_like_value", "bleu"))
                    ),
                    "mode": "api",
                    "details": jour_data
                }
            else:
                return {
                    "date": date,
                    "couleur_tempo": "bleu",
                    "niveau_tension": "normale",
                    "description": "Pas de données disponibles pour cette date",
                    "mode": "api"
                }
                
        except requests.exceptions.Timeout:
            return {
                "error": "Timeout",
                "message": "L'API RTE n'a pas répondu dans le délai imparti",
                "date": date
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
                "message": "Erreur lors de l'appel à l'API RTE",
                "date": date
            }
        except Exception as e:
            return {
                "error": str(e),
                "message": "Erreur inattendue",
                "date": date
            }
    
    def _demo_response(self, date: str) -> Dict[str, Any]:
        """
        Réponse de démonstration basée sur des patterns réalistes
        
        Simule un calendrier Tempo:
        - Hiver (déc-fév): plus de jours rouges
        - Intersaison (mar, nov): jours blancs
        - Été: principalement bleus
        """
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        mois = date_obj.month
        jour = date_obj.day
        jour_semaine = date_obj.weekday()  # 0 = lundi, 6 = dimanche
        
        # Logique de simulation réaliste
        if mois in [12, 1, 2]:  # Hiver
            # Jours rouges plus fréquents en hiver (environ 22 jours/an)
            if jour % 5 == 0 or (jour_semaine == 0 and jour < 20):
                couleur = "rouge"
                tension = "haute"
            elif jour % 3 == 0:
                couleur = "blanc"
                tension = "moyenne"
            else:
                couleur = "bleu"
                tension = "normale"
                
        elif mois in [3, 11]:  # Intersaison
            if jour % 4 == 0:
                couleur = "blanc"
                tension = "moyenne"
            else:
                couleur = "bleu"
                tension = "normale"
                
        else:  # Reste de l'année (printemps/été)
            if jour % 7 == 0:
                couleur = "blanc"
                tension = "moyenne"
            else:
                couleur = "bleu"
                tension = "normale"
        
        return {
            "date": date,
            "couleur_tempo": couleur,
            "niveau_tension": tension,
            "description": self._get_description(tension),
            "mode": "demo",
            "compteur_annuel_estime": self._get_compteur_estime(date_obj)
        }
    
    def _map_couleur_niveau(self, couleur: str) -> str:
        """Mappe la couleur Tempo au niveau de tension"""
        mapping = {
            "rouge": "haute",
            "blanc": "moyenne",
            "bleu": "normale"
        }
        return mapping.get(couleur.lower(), "normale")
    
    def _get_description(self, niveau: str) -> str:
        """Descriptions détaillées des niveaux de tension"""
        descriptions = {
            "haute": (
                "⚠️ JOUR ROUGE - Forte tension sur le réseau électrique. "
                "Tarification maximale EDF. Recommandation forte de réduire "
                "votre consommation, particulièrement aux heures de pointe (6h-22h)."
            ),
            "moyenne": (
                "🟡 JOUR BLANC - Tension modérée sur le réseau. "
                "Tarification intermédiaire. Vigilance conseillée sur "
                "l'utilisation des gros appareils électroménagers."
            ),
            "normale": (
                "🔵 JOUR BLEU - Fonctionnement normal du réseau électrique. "
                "Tarification avantageuse. Pas de restriction particulière "
                "sur la consommation."
            )
        }
        return descriptions.get(niveau, "Information non disponible")
    
    def _get_compteur_estime(self, date_obj: datetime) -> Dict[str, int]:
        """Estime le compteur annuel des jours Tempo (pour mode démo)"""
        # Début de la saison Tempo: 1er septembre
        debut_saison = datetime(date_obj.year if date_obj.month >= 9 else date_obj.year - 1, 9, 1)
        jours_ecoules = (date_obj - debut_saison).days
        
        # Estimation basée sur les quotas annuels réels:
        # 22 jours rouges, 43 jours blancs, 300 jours bleus
        return {
            "rouge_utilises": min(int(jours_ecoules * 22 / 365), 22),
            "blanc_utilises": min(int(jours_ecoules * 43 / 365), 43),
            "rouge_restants": max(22 - int(jours_ecoules * 22 / 365), 0),
            "blanc_restants": max(43 - int(jours_ecoules * 43 / 365), 0)
        }
    
    def get_prevision_semaine(self) -> Dict[str, Any]:
        """
        Récupère les prévisions pour les 7 prochains jours
        
        Returns:
            Dict contenant:
            - periode: Période couverte
            - previsions: Liste des prévisions quotidiennes
            - statistiques: Stats sur la période
        """
        today = datetime.now()
        previsions = []
        compteur_couleurs = {"rouge": 0, "blanc": 0, "bleu": 0}
        
        for i in range(7):
            date = (today + timedelta(days=i)).strftime("%Y-%m-%d")
            jour_info = self.get_jour_tension(date)
            
            if "error" not in jour_info:
                previsions.append(jour_info)
                couleur = jour_info.get("couleur_tempo", "bleu")
                compteur_couleurs[couleur] = compteur_couleurs.get(couleur, 0) + 1
        
        return {
            "periode": f"{previsions[0]['date']} à {previsions[-1]['date']}",
            "previsions": previsions,
            "statistiques": {
                "total_jours": len(previsions),
                "repartition": compteur_couleurs,
                "jours_favorables": compteur_couleurs["bleu"],
                "jours_attention": compteur_couleurs["blanc"] + compteur_couleurs["rouge"]
            },
            "mode": previsions[0].get("mode", "demo") if previsions else "demo"
        }


def format_reponse_rte(data: Dict[str, Any]) -> str:
    """
    Formate la réponse RTE pour un affichage convivial
    
    Args:
        data: Données retournées par JourDeTensionRTE
        
    Returns:
        String formaté pour l'affichage
    """
    # Gestion des erreurs
    if "error" in data:
        return f"❌ **Erreur RTE**: {data['message']}\n\n*Détails: {data['error']}*"
    
    # Format pour prévisions multiples
    if "previsions" in data:
        lignes = [f"# 📅 Prévisions Tempo du {data['periode']}\n"]
        
        for prev in data['previsions']:
            emoji = {
                "rouge": "🔴",
                "blanc": "⚪",
                "bleu": "🔵"
            }.get(prev['couleur_tempo'], "⚫")
            
            jour_obj = datetime.strptime(prev['date'], "%Y-%m-%d")
            jour_nom = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"][jour_obj.weekday()]
            
            lignes.append(
                f"{emoji} **{jour_nom} {prev['date']}** - "
                f"{prev['niveau_tension'].upper()}"
            )
        
        # Ajout des statistiques
        stats = data.get('statistiques', {})
        if stats:
            lignes.append("\n### 📊 Statistiques de la période")
            lignes.append(
                f"- 🔵 Jours bleus (favorables): **{stats.get('jours_favorables', 0)}**"
            )
            lignes.append(
                f"- ⚠️ Jours blanc/rouge: **{stats.get('jours_attention', 0)}**"
            )
        
        # Note sur le mode
        if data.get('mode') == 'demo':
            lignes.append(
                "\n⚠️ *Mode démonstration - "
                "Configurez une clé API RTE pour des données officielles*"
            )
        
        return "\n".join(lignes)
    
    # Format pour une seule date
    else:
        emoji = {
            "rouge": "🔴",
            "blanc": "⚪",
            "bleu": "🔵"
        }.get(data['couleur_tempo'], "⚫")
        
        msg = f"# 📅 Jour Tempo - {data['date']}\n\n"
        msg += f"{emoji} **Couleur**: {data['couleur_tempo'].upper()}\n"
        msg += f"**Niveau**: {data['niveau_tension'].upper()}\n\n"
        msg += f"📝 {data['description']}\n"
        
        # Compteur annuel (mode démo)
        if 'compteur_annuel_estime' in data:
            compteur = data['compteur_annuel_estime']
            msg += f"\n### 📊 Compteur saison (estimé)\n"
            msg += f"- 🔴 Jours rouges: {compteur['rouge_utilises']}/22 (reste: {compteur['rouge_restants']})\n"
            msg += f"- ⚪ Jours blancs: {compteur['blanc_utilises']}/43 (reste: {compteur['blanc_restants']})\n"
        
        # Note mode
        if data.get('mode') == 'demo':
            msg += "\n⚠️ *Mode démonstration*"
        
        return msg


# ═══════════════════════════════════════════════════════════════════════
#                          TESTS
# ═══════════════════════════════════════════════════════════════════════

def run_tests():
    """Exécute les tests de l'agent RTE"""
    print("="*70)
    print("🧪 TESTS AGENT RTE TEMPO")
    print("="*70 + "\n")
    
    agent = JourDeTensionRTE()
    
    # Test 1: Jour actuel
    print("📍 Test 1: Jour actuel")
    print("-"*70)
    result = agent.get_jour_tension()
    print(format_reponse_rte(result))
    print()
    
    # Test 2: Date spécifique
    print("\n📍 Test 2: Date spécifique (25 décembre)")
    print("-"*70)
    result = agent.get_jour_tension("2024-12-25")
    print(format_reponse_rte(result))
    print()
    
    # Test 3: Prévisions 7 jours
    print("\n📍 Test 3: Prévisions 7 jours")
    print("-"*70)
    result = agent.get_prevision_semaine()
    print(format_reponse_rte(result))
    print()
    
    # Test 4: Gestion d'erreur
    print("\n📍 Test 4: Gestion d'erreur (date invalide)")
    print("-"*70)
    result = agent.get_jour_tension("2024-13-45")
    print(format_reponse_rte(result))
    print()
    
    print("="*70)
    print("✅ Tests terminés")
    print("="*70)


if __name__ == "__main__":
    run_tests()
