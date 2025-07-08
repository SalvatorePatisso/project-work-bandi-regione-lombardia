# tools/reader_tools.py
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json
import re

class DocumentAnalysisTool(BaseTool):
    name: str = "Document Analysis Tool"
    description: str = "Analizza documenti di bandi per estrarre informazioni chiave come criteri di eleggibilità, importi, scadenze e requisiti."
    
    def _run(self, document_text: str) -> str:
        """Analizza il documento e estrae informazioni strutturate"""
        
        analysis = {
            "funding_info": self._extract_funding_info(document_text),
            "eligibility_criteria": self._extract_eligibility_criteria(document_text),
            "deadlines": self._extract_deadlines(document_text),
            "requirements": self._extract_requirements(document_text),
            "target_sectors": self._extract_target_sectors(document_text),
            "company_size_requirements": self._extract_company_size(document_text)
        }
        
        return json.dumps(analysis, indent=2, ensure_ascii=False)
    
    def _extract_funding_info(self, text: str) -> Dict[str, Any]:
        """Estrae informazioni sui finanziamenti"""
        funding_patterns = [
            r'(?:€|euro|EUR)\s*[\d.,]+(?:\s*(?:milioni?|miliardi?))?',
            r'(?:importo|finanziamento|contributo).*?(?:€|euro|EUR)\s*[\d.,]+',
            r'(?:fino a|massimo|max).*?(?:€|euro|EUR)\s*[\d.,]+',
            r'[\d.,]+\s*(?:€|euro|EUR)',
            r'(?:budget|dotazione).*?(?:€|euro|EUR)\s*[\d.,]+'
        ]
        
        funding_info = []
        for pattern in funding_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            funding_info.extend(matches)
        
        return {
            "amounts_found": list(set(funding_info)),
            "funding_type": self._determine_funding_type(text)
        }
    
    def _extract_eligibility_criteria(self, text: str) -> List[str]:
        """Estrae criteri di eleggibilità"""
        criteria_keywords = [
            'eleggibil', 'requisit', 'ammissibil', 'criteri',
            'può partecipare', 'possono partecipare', 'destinatari'
        ]
        
        criteria = []
        lines = text.split('\n')
        
        for line in lines:
            for keyword in criteria_keywords:
                if keyword in line.lower():
                    # Estrae la frase completa
                    clean_line = line.strip()
                    if len(clean_line) > 10:
                        criteria.append(clean_line)
        
        return list(set(criteria))[:10]  # Limita a 10 criteri
    
    def _extract_deadlines(self, text: str) -> List[str]:
        """Estrae scadenze e date importanti"""
        date_patterns = [
            r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',
            r'\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}',
            r'(?:scadenza|termine|entro|fino al).*?\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',
            r'(?:scadenza|termine|entro|fino al).*?\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}'
        ]
        
        deadlines = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            deadlines.extend(matches)
        
        return list(set(deadlines))
    
    def _extract_requirements(self, text: str) -> List[str]:
        """Estrae requisiti specifici"""
        requirement_keywords = [
            'deve', 'dovr', 'obbligator', 'necessar', 'richiesto',
            'indispensabil', 'fondamental', 'essenzial'
        ]
        
        requirements = []
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            for keyword in requirement_keywords:
                if keyword in sentence.lower():
                    clean_sentence = sentence.strip()
                    if len(clean_sentence) > 15:
                        requirements.append(clean_sentence)
        
        return list(set(requirements))[:8]  # Limita a 8 requisiti
    
    def _extract_target_sectors(self, text: str) -> List[str]:
        """Estrae settori target"""
        sectors = [
            'tecnologia', 'innovazione', 'digitale', 'ICT', 'software',
            'manifatturiero', 'industria', 'agricoltura', 'turismo',
            'servizi', 'commercio', 'energia', 'ambiente', 'sostenibilità',
            'ricerca', 'sviluppo', 'startup', 'PMI', 'export'
        ]
        
        found_sectors = []
        text_lower = text.lower()
        
        for sector in sectors:
            if sector in text_lower:
                found_sectors.append(sector)
        
        return found_sectors
    
    def _extract_company_size(self, text: str) -> List[str]:
        """Estrae requisiti dimensione aziendale"""
        size_keywords = [
            'startup', 'PMI', 'piccole e medie imprese', 'micro imprese',
            'grande impresa', 'dipendenti', 'fatturato', 'bilancio'
        ]
        
        size_info = []
        text_lower = text.lower()
        
        for keyword in size_keywords:
            if keyword in text_lower:
                size_info.append(keyword)
        
        return size_info
    
    def _determine_funding_type(self, text: str) -> str:
        """Determina il tipo di finanziamento"""
        if 'fondo perduto' in text.lower() or 'contributo a fondo perduto' in text.lower():
            return 'Fondo perduto'
        elif 'prestito' in text.lower() or 'finanziamento agevolato' in text.lower():
            return 'Prestito agevolato'
        elif 'credito d\'imposta' in text.lower():
            return 'Credito d\'imposta'
        else:
            return 'Non specificato'

class BusinessAlignmentTool(BaseTool):
    name: str = "Business Alignment Tool"
    description: str = "Valuta l'allineamento tra un'idea di business e i requisiti di un bando, calcolando un punteggio di compatibilità."
    
    def _run(self, business_idea: str, bando_analysis: str) -> str:
        """Calcola l'allineamento tra business idea e bando"""
        
        try:
            bando_data = json.loads(bando_analysis)
        except:
            bando_data = {"error": "Impossibile parsare l'analisi del bando"}
        
        alignment_score = self._calculate_alignment_score(business_idea, bando_data)
        alignment_explanation = self._generate_alignment_explanation(business_idea, bando_data, alignment_score)
        
        result = {
            "alignment_score": alignment_score,
            "alignment_explanation": alignment_explanation,
            "recommendations": self._generate_recommendations(business_idea, bando_data, alignment_score)
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    def _calculate_alignment_score(self, business_idea: str, bando_data: Dict) -> float:
        """Calcola il punteggio di allineamento (0-100)"""
        score = 0
        max_score = 100
        
        business_lower = business_idea.lower()
        
        # Verifica settori target (30 punti)
        if 'target_sectors' in bando_data:
            sector_matches = 0
            for sector in bando_data['target_sectors']:
                if sector in business_lower:
                    sector_matches += 1
            if sector_matches > 0:
                score += min(30, sector_matches * 10)
        
        # Verifica dimensione aziendale (25 punti)
        if 'company_size_requirements' in bando_data:
            size_keywords = ['startup', 'piccola', 'media', 'PMI']
            for keyword in size_keywords:
                if keyword in business_lower:
                    score += 25
                    break
        
        # Verifica keywords innovative (20 punti)
        innovative_keywords = ['innovazione', 'tecnologia', 'digitale', 'sostenibilità', 'ricerca']
        innovation_matches = sum(1 for keyword in innovative_keywords if keyword in business_lower)
        score += min(20, innovation_matches * 5)
        
        # Verifica presenza di elementi chiave (25 punti)
        key_elements = ['sviluppo', 'mercato', 'prodotto', 'servizio', 'cliente']
        element_matches = sum(1 for element in key_elements if element in business_lower)
        score += min(25, element_matches * 5)
        
        return min(max_score, score)
    
    def _generate_alignment_explanation(self, business_idea: str, bando_data: Dict, score: float) -> str:
        """Genera spiegazione del punteggio di allineamento"""
        
        if score >= 80:
            return f"Ottimo allineamento ({score}%). L'idea di business presenta forte compatibilità con i requisiti del bando."
        elif score >= 60:
            return f"Buon allineamento ({score}%). L'idea di business è compatibile con il bando con alcuni adattamenti."
        elif score >= 40:
            return f"Allineamento moderato ({score}%). L'idea di business richiede modifiche significative per adattarsi al bando."
        else:
            return f"Allineamento limitato ({score}%). L'idea di business presenta poca compatibilità con i requisiti del bando."
    
    def _generate_recommendations(self, business_idea: str, bando_data: Dict, score: float) -> List[str]:
        """Genera raccomandazioni basate sull'allineamento"""
        
        recommendations = []
        
        if score >= 70:
            recommendations.append("Procedere con la candidatura al bando")
            recommendations.append("Preparare la documentazione richiesta")
            recommendations.append("Verificare tutti i requisiti specifici")
        else:
            recommendations.append("Valutare modifiche all'idea di business per aumentare l'allineamento")
            recommendations.append("Considerare bandi alternativi più adatti")
            recommendations.append("Consultare esperti per ottimizzare la proposta")
        
        return recommendations