import type { Severity } from "./types.ts";
export interface RuleDoc {
    id: string;
    severity: Severity;
    /** what it catches, in one line */
    catches: string;
    /** the option on analyze(plan, rules) that moves the threshold, if any */
    option?: string;
}
export declare const RULES: readonly RuleDoc[];
export declare const ruleById: (id: string) => RuleDoc | undefined;
