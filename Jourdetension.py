#!/usr/bin/env python
"""
jourdetension_agent.py
─────────────────────────────
Agent pour récupérer les jours de tension RTE
"""
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class JourDeTensionRTE:
    """Agent pour interroger l'API RTE sur les jours de tension électrique"""
    
    BASE_URL = "https://digital.iservices.rte-france.com/open_api/tempo_like_supply_contract/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize avec une clé API RTE (optionnelle pour démo)
        Pour obtenir une vraie clé: https://data.rte-france.com/
        """
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}" if api_key else "",
            "Content-Type": "application/json"
        }
    
    def get_jour_tension(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère le statut de tension pour une date donnée
        
        Args:
            date: Date au format 'YYYY-MM-DD'. Si None, utilise aujourd'hui
            
        Returns:
            Dict contenant les informations de tension
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Mode démo si pas de clé API
        if not self.api_key:
            return self._demo_response(date)
        
        try:
            # Appel API réel (nécessite authentification)
            response = requests.get(
                f"{self.BASE_URL}/tempo_like_calendars",
                headers=self.headers,
                params={"start_date": date, "end_date": date},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "error": str(e),
                "message": "Erreur lors de l'appel à l'API RTE"
            }
    
    def _demo_response(self, date: str) -> Dict[str, Any]:
        """Réponse de démonstration basée sur des patterns réalistes"""
        # Logique simplifiée : jours de tension en hiver
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        mois = date_obj.month
        
        # Hiver = plus de risques de tension
        if mois in [12, 1, 2]:
            couleur = "rouge" if date_obj.day % 7 == 0 else "orange"
            tension = "haute" if couleur == "rouge" else "moyenne"
        elif mois in [3, 11]:
            couleur = "orange"
            tension = "moyenne"
        else:
            couleur = "vert"
            tension = "normale"
        
        return {
            "date": date,
            "couleur_tempo": couleur,
            "niveau_tension": tension,
            "description": self._get_description(tension),
            "mode": "demo"
        }
    
    def _get_description(self, niveau: str) -> str:
        """Descriptions des niveaux de tension"""
        descriptions = {
            "haute": "Forte tension sur le réseau électrique. "
                    "Recommandation de réduire sa consommation aux heures de pointe.",
            "moyenne": "Tension modérée sur le réseau. "
                      "Vigilance conseillée sur les gros appareils.",
            "normale": "Pas de tension particulière sur le réseau électrique."
        }
        return descriptions.get(niveau, "Information non disponible")
    
    def get_prevision_semaine(self) -> Dict[str, Any]:
        """Récupère les prévisions pour les 7 prochains jours"""
        today = datetime.now()
        previsions = []
        
        for i in range(7):
            date = (today + timedelta(days=i)).strftime("%Y-%m-%d")
            jour_info = self.get_jour_tension(date)
            previsions.append(jour_info)
        
        return {
            "periode": f"{previsions[0]['date']} à {previsions[-1]['date']}",
            "previsions": previsions
        }


def format_reponse_rte(data: Dict[str, Any]) -> str:
    """Formate la réponse RTE pour le chatbot"""
    if "error" in data:
        return f"❌ {data['message']}: {data['error']}"
    
    if "previsions" in data:
        # Format pour prévisions multiples
        lignes = [f"📅 Prévisions du {data['periode']}:\n"]
        for prev in data['previsions']:
            emoji = {"rouge": "🔴", "orange": "🟠", "vert": "🟢"}.get(prev['couleur_tempo'], "⚪")
            lignes.append(
                f"{emoji} {prev['date']}: {prev['niveau_tension'].upper()} - {prev['description']}"
            )
        return "\n".join(lignes)
    else:
        # Format pour une seule date
        emoji = {"rouge": "🔴", "orange": "🟠", "vert": "🟢"}.get(data['couleur_tempo'], "⚪")
        msg = f"📅 Tension RTE pour le {data['date']}:\n"
        msg += f"{emoji} Niveau: {data['niveau_tension'].upper()}\n"
        msg += f"📝 {data['description']}"
        if data.get('mode') == 'demo':
            msg += "\n\n⚠️ Mode démonstration (configurez une clé API RTE pour des données réelles)"
        return msg


# Point d'entrée pour test
if __name__ == "__main__":
    agent = JourDeTensionRTE()
    
    print("Test 1: Jour actuel")
    print(format_reponse_rte(agent.get_jour_tension()))
    
    print("\n" + "="*60 + "\n")
    
    print("Test 2: Prévisions 7 jours")
    print(format_reponse_rte(agent.get_prevision_semaine()))