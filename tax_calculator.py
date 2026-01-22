class SimpleTaxCalculator:
    def __init__(self):
        # 2024 US Federal Tax Brackets (for single filers)
        self.tax_brackets = [
            (11000, 0.10),      # 10% on income up to $11,000
            (44725, 0.12),      # 12% on income $11,001 to $44,725
            (95375, 0.22),      # 22% on income $44,726 to $95,375
            (182050, 0.24),     # 24% on income $95,376 to $182,050
            (231250, 0.32),     # 32% on income $182,051 to $231,250
            (578125, 0.35),     # 35% on income $231,251 to $578,125
            (float('inf'), 0.37) # 37% on income over $578,125
        ]
        
        # Standard deduction for 2024
        self.standard_deduction = {
            'single': 13850,
            'married_joint': 27700,
            'married_separate': 13850,
            'head_of_household': 20800
        }
    
    def calculate_federal_tax(self, annual_income, filing_status='single', deductions=None):
        """Calculate federal income tax"""
        
        # Use standard deduction if no custom deductions provided
        if deductions is None:
            deductions = self.standard_deduction.get(filing_status, 13850)
        
        # Calculate taxable income
        taxable_income = max(0, annual_income - deductions)
        
        if taxable_income == 0:
            return {
                'gross_income': annual_income,
                'deductions': deductions,
                'taxable_income': 0,
                'federal_tax': 0,
                'effective_rate': 0,
                'marginal_rate': 0,
                'after_tax_income': annual_income
            }
        
        # Calculate tax using brackets
        tax_owed = 0
        previous_bracket = 0
        marginal_rate = 0
        
        for bracket_limit, rate in self.tax_brackets:
            if taxable_income > previous_bracket:
                # Calculate tax for this bracket
                taxable_in_bracket = min(taxable_income, bracket_limit) - previous_bracket
                tax_owed += taxable_in_bracket * rate
                marginal_rate = rate
                
                if taxable_income <= bracket_limit:
                    break
                    
                previous_bracket = bracket_limit
        
        effective_rate = (tax_owed / annual_income) * 100 if annual_income > 0 else 0
        after_tax_income = annual_income - tax_owed
        
        return {
            'gross_income': annual_income,
            'deductions': deductions,
            'taxable_income': taxable_income,
            'federal_tax': round(tax_owed, 2),
            'effective_rate': round(effective_rate, 2),
            'marginal_rate': round(marginal_rate * 100, 1),
            'after_tax_income': round(after_tax_income, 2)
        }
    
    def calculate_state_tax(self, annual_income, state_rate=0.05):
        """Simple state tax calculation (flat rate)"""
        return annual_income * state_rate
    
    def calculate_payroll_taxes(self, annual_income):
        """Calculate Social Security and Medicare taxes"""
        # Social Security: 6.2% up to $160,200 (2024 limit)
        ss_limit = 160200
        ss_rate = 0.062
        
        # Medicare: 1.45% on all income + 0.9% additional on income over $200,000
        medicare_rate = 0.0145
        additional_medicare_threshold = 200000
        additional_medicare_rate = 0.009
        
        # Social Security tax
        ss_taxable = min(annual_income, ss_limit)
        ss_tax = ss_taxable * ss_rate
        
        # Medicare tax
        medicare_tax = annual_income * medicare_rate
        if annual_income > additional_medicare_threshold:
            additional_medicare = (annual_income - additional_medicare_threshold) * additional_medicare_rate
            medicare_tax += additional_medicare
        
        total_payroll_tax = ss_tax + medicare_tax
        
        return {
            'social_security_tax': round(ss_tax, 2),
            'medicare_tax': round(medicare_tax, 2),
            'total_payroll_tax': round(total_payroll_tax, 2)
        }
    
    def calculate_total_taxes(self, annual_income, filing_status='single', 
                            state_rate=0.05, custom_deductions=None):
        """Calculate all taxes combined"""
        
        # Federal tax
        federal_result = self.calculate_federal_tax(annual_income, filing_status, custom_deductions)
        
        # State tax (simplified)
        state_tax = self.calculate_state_tax(annual_income, state_rate)
        
        # Payroll taxes
        payroll_result = self.calculate_payroll_taxes(annual_income)
        
        # Total taxes
        total_tax = (federal_result['federal_tax'] + 
                    state_tax + 
                    payroll_result['total_payroll_tax'])
        
        net_income = annual_income - total_tax
        total_tax_rate = (total_tax / annual_income * 100) if annual_income > 0 else 0
        
        return {
            'gross_income': annual_income,
            'federal_tax': federal_result['federal_tax'],
            'state_tax': round(state_tax, 2),
            'payroll_taxes': payroll_result['total_payroll_tax'],
            'total_tax': round(total_tax, 2),
            'net_income': round(net_income, 2),
            'total_tax_rate': round(total_tax_rate, 2),
            'federal_details': federal_result,
            'payroll_details': payroll_result
        }

# Example usage and testing
if __name__ == "__main__":
    calculator = SimpleTaxCalculator()
    
    # Test with different incomes
    test_incomes = [30000, 50000, 75000, 100000, 150000]
    
    print("=== TAX CALCULATION EXAMPLES ===\n")
    
    for income in test_incomes:
        result = calculator.calculate_total_taxes(income, 'single', 0.05)
        
        print(f"Annual Income: ${income:,}")
        print(f"Federal Tax: ${result['federal_tax']:,}")
        print(f"State Tax (5%): ${result['state_tax']:,}")
        print(f"Payroll Tax: ${result['payroll_taxes']:,}")
        print(f"Total Tax: ${result['total_tax']:,}")
        print(f"Net Income: ${result['net_income']:,}")
        print(f"Total Tax Rate: {result['total_tax_rate']}%")
        print("-" * 40)