import { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="border border-warn/40 bg-warn/10 p-4 text-sm">
          <p className="font-medium text-warn">Something broke in this panel.</p>
          <p className="mt-1 text-muted">{this.state.error.message}</p>
        </div>
      );
    }
    return this.props.children;
  }
}
