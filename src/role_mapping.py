"""
Module for mapping personality types to roles
"""
class RoleMapper:
    def __init__(self):
        self.role_mapping = {
            'INTJ': 'Strategist',
            'INTP': 'Architect',
            'ENTJ': 'Commander',
            'ENTP': 'Debater',
            'INFJ': 'Advocate',
            'INFP': 'Mediator',
            'ENFJ': 'Protagonist',
            'ENFP': 'Campaigner',
            'ISTJ': 'Logistician',
            'ISFJ': 'Defender',
            'ESTJ': 'Executive',
            'ESFJ': 'Consul',
            'ISTP': 'Virtuoso',
            'ISFP': 'Adventurer',
            'ESTP': 'Entrepreneur',
            'ESFP': 'Entertainer'
        }
    
    def get_role(self, personality_type):
        """Get role based on personality type"""
        return self.role_mapping.get(personality_type, 'Unknown')
    
    def get_career_suggestions(self, personality_type):
        """Get career suggestions based on personality type"""
        # TODO: Implement more detailed career suggestions
        return [self.get_role(personality_type)] 